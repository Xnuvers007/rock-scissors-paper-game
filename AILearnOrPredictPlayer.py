import cv2
import mediapipe as mp
import numpy as np
import random
import time
from collections import deque, Counter

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

player_score = 0
ai_score = 0

history = deque(maxlen=10)

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

def predict_player_move():
    if not history:
        return random.choice(["Rock", "Paper", "Scissors"])
    
    count = Counter(history)
    predicted = count.most_common(1)[0][0]

    counter_moves = {
        "Rock": "Paper",
        "Paper": "Scissors",
        "Scissors": "Rock"
    }
    return counter_moves.get(predicted, random.choice(["Rock", "Paper", "Scissors"]))

def get_winner(player, ai):
    if player == ai:
        return "Draw"
    elif (player == "Rock" and ai == "Scissors") or \
         (player == "Paper" and ai == "Rock") or \
         (player == "Scissors" and ai == "Paper"):
        return "Player"
    else:
        return "AI"

cap = cv2.VideoCapture(0)

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
                ai_move = predict_player_move()
                winner = get_winner(player_move, ai_move)

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

    cv2.imshow("Rock Paper Scissors - Smart AI", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
