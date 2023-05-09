import cv2
import numpy as np

# center_pt_x, center_pt_y, fake_dis, online_ids[idx], tlwh
def bird_eye_transform(person_pts, M, H=1000, W=500):
    img = np.zeros(((H, W, 3)), np.uint8)
    if len(person_pts) != 0:
        person_pts = np.asarray(person_pts, dtype=np.float32)
        print(person_pts)

        transformed_pt = cv2.perspectiveTransform(
                person_pts[:, :2].reshape(1, -1, 2), M,
            )
        # if transformed_pt is not None:
        for i, pt in enumerate(transformed_pt[0]):
            # print(pt)
            cv2.circle(img, tuple(map(int, pt)), 3, (0, 255, 0), -1)
            cv2.putText(img, str(int(person_pts[i][2])), tuple(map(int, pt)), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (255, 255, 0), 1, cv2.LINE_AA)

    return img