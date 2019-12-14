import cv2
import numpy as np
from copy import deepcopy as cp
import classify
# Load camera 0
cam = cv2.VideoCapture(0)

# Initialize some windows for showing different stages of processing
cv2.namedWindow("cameraFeed")
cv2.namedWindow("paperFeed")
cv2.namedWindow("signFeed")

HEIGHT = 480
WIDTH = 640

def is_white(pixel, delta=100):
    # Check for each value in BGR if it is above the threshold delta
    # If all are higher return true
    return pixel[0] + delta >= 255 and pixel[1] + delta >= 255 and pixel[2] + delta >= 255


def simple_crop(frame, l, r, u, d):
    # Crop frame based on 4 corners
    return frame[u:d, l:r]


def crop_frame(frame, l, r, u, d):
    # If no dimensions are specified, crop maximum amount.
    width = 200
    height = 100

    # If the y of the downmost coordinate - y of leftmost coordinate
    # is more than the width // 2, the dowmost point is the rightbottom point
    # and the leftmost point is the leftbottom point.
    if d[0] - l[0] >= width // 2:
        points = np.float32([u, r, l, d])
    else:
        points = np.float32([l, u, d, r])

    # An array with 4 coordinates for every corner to warp to
    warp = np.float32([
        [0, 0],
        [width, 0],
        [0, height],
        [width, height]
    ])

    # Warp the points coordinates to the warp coordinates and return the new view.
    matrix = cv2.getPerspectiveTransform(points, warp)
    return cv2.warpPerspective(frame, matrix, (width, height))


def generate_noise(shape):
    # Generate a random color matrix based on sign view shape (should be 32x32x1)
    b = np.array(np.random.normal(loc=1, scale=0.5, size=[shape[0], shape[1], 1]) * 255, dtype=np.uint8)
    g = np.array(np.random.normal(loc=1, scale=0.5, size=[shape[0], shape[1], 1]) * 255, dtype=np.uint8)
    r = np.array(np.random.normal(loc=1, scale=0.5, size=[shape[0], shape[1], 1]) * 255, dtype=np.uint8)
    return np.concatenate((b, g, r), axis=2)


def fill_background(frame, to_fill):
    if np.shape(frame) != np.shape(to_fill):
        print('fill_background: shapes don\'t match')
        return

    height = len(frame)
    width = len(frame[0])

    # Start filling from 2, 2
    x, y = (2, 2)
    # Push that pixel to the queue.
    q = [[x, y]]

    # Create matrix filled with zeros (0 means unvisited)
    viz = [[False for j in range(width)] for i in range(height)]
    viz[x][y] = True

    # If the queue is not empty
    while len(q):
        # Save x and y of first element of queue.
        x, y = q[0]
        # Pop the first element of queue.
        q = q[1:]

        # In the following lines, we try to add more pixels to the queue.
        # The first check: if the value is within the view.
        # The second check: if the value has not been added to the queue yet (not viz(ited)).
        # The third check: if the value is actually white.
        # If all are true, add value to the queue and mark it visited.
        if x > 0 and not viz[x - 1][y] and is_white(frame[x - 1][y]):
            viz[x - 1][y] = True
            q.append([x - 1, y])
        if x < height - 1 and not viz[x + 1][y] and is_white(frame[x + 1][y]):
            viz[x + 1][y] = True
            q.append([x + 1, y])
        if y > 0 and not viz[x][y - 1] and is_white(frame[x][y - 1]):
            viz[x][y - 1] = True
            q.append([x, y - 1])
        if y < width - 1 and not viz[x][y + 1] and is_white(frame[x][y + 1]):
            viz[x][y + 1] = True
            q.append([x, y + 1])

    # Go through every pixel in frame.
    # If that cell was marked visited,
    # copy the value of to_fill to frame.
    for i in range(len(frame)):
        for j in range(len(frame[i])):
            if viz[i][j]:
                frame[i][j] = to_fill[i][j]

    return frame


def fill_sign(frame, skip=0, padding=20):
    height = len(frame)
    width = len(frame[0])

    # Start filling from 2, 2
    x, y = (2, 2)
    # Push that pixel to the queue.
    q = [[x, y]]

    # Create matrix filled with zeros (0 means unvisited)
    viz = [[False for j in range(width)] for i in range(height)]
    viz[x][y] = True

    #
    l = width
    r = 0
    u = height
    d = 0

    # If the queue is not empty
    while len(q):
        # Save x and y of first element of queue.
        x, y = q[0]
        # Pop the first element of queue.
        q = q[1:]

        # In the following lines, we try to add more pixels to the queue.
        # The first check: if the value -/+ skip is within the view.
        # The second check: if the value -/+ skip has not been added to the queue yet (not viz(ited)).
        # If all are true, 
            # If the value -/+ is white,
            # add value -/+ skip to the queue and mark it visited.
            # Else if the current x or y is nearer to the center than the saved coordinates, overwrite the saved coordinates.
        if x > skip and not viz[x - skip - 1][y]:
            if is_white(frame[x - skip - 1][y]):
                q.append([x - skip - 1, y])
                viz[x - skip - 1][y] = True
            else:
                if l > y:
                    l = y
                if r < y:
                    r = y
                if u > x - skip - 1:
                    u = x - skip - 1
                if d < x - skip - 1:
                    d = x - skip - 1
        if y > skip and not viz[x][y - skip - 1]:
            if is_white(frame[x][y - skip - 1]):
                q.append([x, y - skip - 1])
                viz[x][y - skip - 1] = True
            else:
                if l > y - skip - 1:
                    l = y - skip - 1
                if r < y - skip - 1:
                    r = y - skip - 1
                if u > x:
                    u = x
                if d < x:
                    d = x
        if x < height - skip - 1 and not viz[x + skip + 1][y]:
            if is_white(frame[x + skip + 1][y]):
                q.append([x + skip + 1, y])
                viz[x + skip + 1][y] = True
            else:
                if l > y:
                    l = y
                if r < y:
                    r = y
                if u > x + skip + 1:
                    u = x + skip + 1
                if d < x + skip + 1:
                    d = x + skip + 1
        if y < width - skip - 1 and not viz[x][y + skip + 1]:
            if is_white(frame[x][y + skip + 1]):
                q.append([x, y + skip + 1])
                viz[x][y + skip + 1] = True
            else:
                if l > y + skip + 1:
                    l = y + skip + 1
                if r < y + skip + 1:
                    r = y + skip + 1
                if u > x:
                    u = x
                if d < x:
                    d = x

    # Return the frame and remove some padding.
    # Note: Also check if l - padding is less than zero (out of frame),
    # whereafter you just take the min or max croppable.
    return simple_crop(frame, max(0, l - padding), min(width - 1, r + padding), max(0, u - padding), min(height - 1, d + padding))

def fill_pixel(frame, x, y, radius=1, color=[0, 0, 0]):
    # Draw color to frame at x, y and surrounding pixels in radius
    for i in range(x - radius, x + radius + 1):
        for j in range(y - radius, y + radius + 1):
            frame[i][j] = color
    return frame

def fill_paper(frame, skip=0, amount=None, padding=0):
    # Start in the middle.
    x = HEIGHT // 2
    y = WIDTH // 2

    # Push middle pixel to the queue.
    q = [[x, y]]

    # Create matrix filled with zeros (0 means unvisited)
    viz = [[False for j in range(WIDTH)] for i in range(HEIGHT)]
    viz[x][y] = True

    # Initialize coordinates of following points.
    # We keep track of this so that we know where
    # the corners of the frame are.

    # Just the x or y values.
    leftmost = WIDTH
    rightmost = 0
    upmost = HEIGHT
    downmost = 0

    # Actual coordinates.
    lpoint = None
    rpoint = None
    upoint = None
    dpoint = None


    # Keep track of how many pixels are pushed.
    loop_cnt = 0

    # If the queue is not empty
    while len(q) and amount:
        # Save x and y of first element of queue.
        x, y = q[0]
        # Pop the first element of queue.
        q = q[1:]

        # Check if the current x or y is more to the side than the saved coordinates.
        # If so, overwrite with current coordinate. (- padding).
        # We optionally do some negative padding to remove the black borders from the view.
        if leftmost > y:
            leftmost = y
            lpoint = [y+padding, x]
        if rightmost < y:
            rightmost = y
            rpoint = [y-padding, x]
        if upmost > x:
            upmost = x
            upoint = [y, x+padding]
        if downmost < x:
            downmost = x
            dpoint = [y, x-padding]

        # Draw a black pixel to the view
        frame = fill_pixel(frame, x, y)

        # In the following lines, we try to add more pixels to the queue.
        # The first check: if the value -/+ skip is within the view.
        # The second check: if the value -/+ skip has not been added to the queue yet (not viz(ited)).
        # The third check: if the value -/+ skip is actually white.
        # If all are true, add value -/+ skip to the queue and mark it visited
        if x > skip and not viz[x - skip - 1][y] and is_white(frame[x - skip - 1][y]):
            q.append([x - skip - 1, y])
            viz[x - skip - 1][y] = True
        if y > skip and not viz[x][y - skip - 1] and is_white(frame[x][y - skip - 1]):
            q.append([x, y - skip - 1])
            viz[x][y - skip - 1] = True
        if x < HEIGHT - skip - 1 and not viz[x + skip + 1][y] and is_white(frame[x + skip + 1][y]):
            q.append([x + skip + 1, y])
            viz[x + skip + 1][y] = True
        if y < WIDTH - skip - 1 and not viz[x][y + skip + 1] and is_white(frame[x][y + skip + 1]):
            q.append([x, y + skip + 1])
            viz[x][y + skip + 1] = True

        # If there is a limit given
        # Decrease the amount until 0 is reached
        if amount is not None:
            amount -= 1
            if amount < 0:
                break
        loop_cnt += 1

    return frame, viz, (lpoint, rpoint, upoint, dpoint), loop_cnt

if __name__ == '__main__':
    animation_cnt = 1
    
    classify_counter = 0
    max_classify = 10
    res = np.zeros(43)

    while True:
        # Save current camera feed into original.
        ret, frame = cam.read()
        original = cp(frame)

        # Fill white pixels within border.
        frame, viz, points, loop_cnt = fill_paper(frame, 3, animation_cnt * 40)

        # If the BFS radius is more than 10,
        # Crop, rotate, and warp the image flat to 200,100.
        # Else, hide the view.
        if loop_cnt > 10:
            l, r, u, d = points
            paper_view = crop_frame(original, l, r, u, d)
        else:
            paper_view = None

        # If BFS radius has not reached more than five, reset the animation.
        if animation_cnt > 6 and loop_cnt < 5:
            animation_cnt = 0
        animation_cnt += 1

        # Display the camera feed and draw red pointer pixel in the middle.
        cv2.imshow("cameraFeed", fill_pixel(frame, HEIGHT // 2, WIDTH // 2, radius=2, color=[0, 0, 255]))

        # Returns the matrix format (should be 200, 100, 3)
        view_shape = np.shape(paper_view)
        
        # If a shape of that format has been found continue
        if paper_view is not None and view_shape[0] > 0 and view_shape[1] > 0:  
            # Show the cropped paper view
            cv2.imshow("paperFeed", paper_view)

            # Fill the sign with black dots and crop to sign.
            sign_view = fill_sign(paper_view, skip=3)

            # Returns the matrix format (should be w, h, 3).
            sign_shape = np.shape(sign_view)

            # If a shape of that format has been found continue.
            if sign_shape[0] > 0 and sign_shape[1] > 0:
                # Resize the sign view to 32x32 as the network takes 32x32.
                final_image = cv2.resize(sign_view, (32, 32))

                # Generate noise and add that noise to the background of the final image.
                noise = generate_noise(np.shape(final_image))
                final_image = fill_background(final_image, noise)

                # Show the final image.
                cv2.imshow("signFeed", final_image)

                # Grayscale the image.
                final_image = cv2.cvtColor(final_image, cv2.COLOR_BGR2GRAY)
                final_image = np.reshape(final_image, (32, 32, 1))

                # Save result of classification and increment the counter                
                res += classify.classify(final_image/255)[0]
                classify_counter += 1

                if classify_counter > max_classify:
                    # Get the most occuring classification result and print his label.
                    print(classify.labels[np.argmax(res)])
                    # Reset counter & res.
                    classify_counter = 0
                    res = np.zeros(43)
        
        # Keep the camera view open until ESC is pressed.
        k = cv2.waitKey(1)
        if k % 256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break
    
    cam.release()
    cv2.destroyAllWindows()
