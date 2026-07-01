"""
Generate a synthetic reference/scene image pair to demo the pipeline.

- reference.png : a clean "board" with a few components.
- scene.png     : the same board rotated + translated, with an injected
                  defect (a foreign red object) and a missing component.

This lets align.py run out of the box, with no external dataset.
Replace these with your own images anytime.
"""
import os
import cv2
import numpy as np

os.makedirs("images", exist_ok=True)


def make_board():
    rng = np.random.default_rng(0)
    img = np.full((400, 500, 3), 90, np.uint8)          # gray board
    # dense speckle texture -> gives SIFT plenty of stable keypoints
    for _ in range(900):
        x, y = int(rng.integers(0, 500)), int(rng.integers(0, 400))
        r = int(rng.integers(1, 3))
        c = int(rng.integers(30, 230))
        cv2.circle(img, (x, y), r, (c, c, c), -1)
    # a few distinct "components" drawn on top
    cv2.rectangle(img, (60, 60), (160, 120), (200, 200, 200), -1)
    cv2.rectangle(img, (300, 80), (420, 160), (180, 180, 180), -1)
    cv2.circle(img, (120, 260), 35, (0, 200, 0), -1)
    cv2.circle(img, (350, 280), 30, (0, 180, 220), -1)
    cv2.rectangle(img, (200, 200), (260, 320), (150, 150, 150), -1)
    return img


ref = make_board()
cv2.imwrite("images/reference.png", ref)

# build a DEFECTIVE board (2 defects), THEN rotate + translate it -> scene
h, w = ref.shape[:2]
defective = ref.copy()
cv2.rectangle(defective, (60, 60), (160, 120), (90, 90, 90), -1)   # defect 1: missing component
cv2.circle(defective, (350, 300), 18, (0, 0, 255), -1)            # defect 2: foreign red object

M = cv2.getRotationMatrix2D((w / 2, h / 2), 12, 1.0)
M[0, 2] += 25
M[1, 2] += 15
scene = cv2.warpAffine(defective, M, (w, h), borderValue=(90, 90, 90))
cv2.imwrite("images/scene.png", scene)

print("Wrote images/reference.png and images/scene.png")
