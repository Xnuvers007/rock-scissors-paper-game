import cv2
import mediapipe as mp
import numpy as np
import random
import time
from collections import deque, defaultdict

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

player_score = 0
ai_score = 0
history = deque(maxlen=20)
transition_table = defaultdict(lambda: defaultdict(int))

def detect_gesture(landmarks):
    fingers = []
    fingers.append(landmarks[4].x < landmarks[3].x)
    for tip_id in [8, 12, 16, 20]:
        fingers.append(landmarks[tip_id].y < landmarks[tip_id - 2].y)
    if fingers == [False, False, False, False, False]:
        return "Rock"
    elif fingers == [True, True, True, True, True]:
        return "Paper"
    elif fingers[1] and fingers[2] and not fingers[3] and not fingers[4]:
        return "Scissors"
    else:
        return "Unknown"

def predict_player_move_markov():
    if len(history) < 2:
        return random.choice(["Rock", "Paper", "Scissors"])
    last_move = history[-1]
    next_moves = transition_table[last_move]
    if not next_moves:
        return random.choice(["Rock", "Paper", "Scissors"])
    predicted_next = max(next_moves, key=next_moves.get)
    counter_moves = {"Rock": "Paper", "Paper": "Scissors", "Scissors": "Rock"}
    return counter_moves.get(predicted_next, random.choice(["Rock", "Paper", "Scissors"]))

def get_winner(player, ai):
    if player == ai:
        return "Draw"
    elif (player == "Rock" and ai == "Scissors") or \
         (player == "Paper" and ai == "Rock") or \
         (player == "Scissors" and ai == "Paper"):
        return "Player"
    else:
        return "AI"

cap = cv2.VideoCapture(1)

last_time = time.time()
show_result = False
result = ""
player_move = ""
ai_move = ""

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result_hands = hands.process(rgb)

    if result_hands.multi_hand_landmarks:
        for hand_landmarks in result_hands.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            landmarks = hand_landmarks.landmark
            player_move = detect_gesture(landmarks)
            if time.time() - last_time > 3 and player_move != "Unknown":
                ai_move = predict_player_move_markov()
                winner = get_winner(player_move, ai_move)
                if history:
                    prev_move = history[-1]
                    transition_table[prev_move][player_move] += 1
                history.append(player_move)
                if winner == "Player":
                    player_score += 1
                elif winner == "AI":
                    ai_score += 1
                result = f"{winner} wins!" if winner != "Draw" else "It's a draw!"
                show_result = True
                last_time = time.time()
                break

    cv2.putText(frame, f"Player: {player_score} | AI: {ai_score}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    if show_result:
        cv2.putText(frame, f"You: {player_move}", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(frame, f"AI: {ai_move}", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
        cv2.putText(frame, result, (10, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        if time.time() - last_time > 2:
            show_result = False

    cv2.putText(frame, "Show gesture: Rock, Paper, or Scissors", (10, 460),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)

    cv2.imshow("Rock Paper Scissors - Genius AI", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
