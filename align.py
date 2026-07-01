"""
Part inspection by feature-based alignment (SIFT + FLANN + RANSAC).

Pipeline
--------
1. Detect SIFT keypoints/descriptors on a reference and a scene image.
2. Match descriptors (FLANN) and keep good matches (Lowe's ratio test).
3. Estimate the scene -> reference homography with RANSAC.
4. Warp the scene into the reference frame.
5. Compare (absolute difference) to highlight defects / missing parts.

Usage
-----
    python align.py                         # uses images/reference.png + images/scene.png
    python align.py --ref R.png --scene S.png
Results are written to outputs/.
"""
import argparse
import os
import cv2
import numpy as np


def load(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {path}")
    return img, cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def match_features(ref_gray, scene_gray, ratio=0.75):
    sift = cv2.SIFT_create()
    kp_ref, des_ref = sift.detectAndCompute(ref_gray, None)
    kp_scene, des_scene = sift.detectAndCompute(scene_gray, None)

    # FLANN with a KD-tree (SIFT produces float descriptors)
    flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
    knn = flann.knnMatch(des_ref, des_scene, k=2)

    # Lowe's ratio test: keep a match only if clearly better than the 2nd best
    good = [m for m, n in knn if m.distance < ratio * n.distance]
    return kp_ref, kp_scene, good


def estimate_homography(kp_ref, kp_scene, good, min_matches=10):
    if len(good) < min_matches:
        raise RuntimeError(f"Not enough good matches: {len(good)} < {min_matches}")
    pts_ref = np.float32([kp_ref[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    pts_scene = np.float32([kp_scene[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    # H maps SCENE coordinates -> REFERENCE coordinates
    H, mask = cv2.findHomography(pts_scene, pts_ref, cv2.RANSAC, 5.0)
    return H, int(mask.sum())


def highlight_defects(ref_bgr, aligned_bgr, valid_mask=None, thresh=45, min_area=120):
    diff = cv2.absdiff(cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2GRAY),
                       cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2GRAY))
    _, mask = cv2.threshold(diff, thresh, 255, cv2.THRESH_BINARY)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    if valid_mask is not None:                 # ignore the warp's black borders
        mask = cv2.bitwise_and(mask, valid_mask)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out, n = ref_bgr.copy(), 0
    for c in contours:
        if cv2.contourArea(c) < min_area:
            continue
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 0, 255), 2)
        n += 1
    return out, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="images/reference.png")
    ap.add_argument("--scene", default="images/scene.png")
    ap.add_argument("--out", default="outputs")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    ref_bgr, ref_gray = load(args.ref)
    scene_bgr, scene_gray = load(args.scene)

    kp_ref, kp_scene, good = match_features(ref_gray, scene_gray)
    print(f"Good matches      : {len(good)}")

    H, inliers = estimate_homography(kp_ref, kp_scene, good)
    print(f"RANSAC inliers    : {inliers}")

    h, w = ref_gray.shape
    aligned = cv2.warpPerspective(scene_bgr, H, (w, h))

    # region actually covered by the warped scene (drop black borders)
    valid = cv2.warpPerspective(np.full(scene_gray.shape, 255, np.uint8), H, (w, h))
    valid = cv2.erode(valid, np.ones((9, 9), np.uint8))

    defects, n = highlight_defects(ref_bgr, aligned, valid)
    print(f"Defects detected  : {n}")

    matches_vis = cv2.drawMatches(
        ref_bgr, kp_ref, scene_bgr, kp_scene, good, None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imwrite(os.path.join(args.out, "matches.png"), matches_vis)
    cv2.imwrite(os.path.join(args.out, "aligned.png"), aligned)
    cv2.imwrite(os.path.join(args.out, "defects.png"), defects)
    print(f"Saved results to  : {args.out}/")


if __name__ == "__main__":
    main()
