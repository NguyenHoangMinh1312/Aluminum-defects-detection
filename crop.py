import cv2

def crop(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image not found. Check your path.")
    img_h, img_w = image.shape[:2]

    # convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # gaussian blur
    blur = cv2.GaussianBlur(gray, (5, 5), 1)

    # canny edge detection
    edges = cv2.Canny(blur, 50, 150)

    # morphological closing
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # find contours
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        raise ValueError("No contours found in image.")

    # get largest contour
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < 50:
        raise ValueError("Found contour is too small (likely noise).")

    # get bounding box
    x, y, w, h = cv2.boundingRect(contour)

    padding = 50
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(img_w, x + w + padding)
    y2 = min(img_h, y + h + padding)

    # crop from the original image
    cropped = image[y1:y2, x1:x2]

    return cropped

if __name__ == "__main__":
    image_path = "datasets_origin/normal/normal.bmp"

    res = crop(image_path)
    cv2.imshow("Result", res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()