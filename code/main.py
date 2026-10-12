import cv2
import numpy as np
import os
import pandas as pd
from ultralytics import YOLO

model = YOLO('yolov8s.pt')#importing the model
# getting right path of colors.csv.txt file in cde script
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, "colors.csv.txt")

index = ["Color", "color_name", "hex", "R", "G", "B"]
csv = pd.read_csv(csv_path, names=index, header=None)#reading the csv file
# print(csv.head(10))
unique_labels = set() # Set to store unique labels
# variables for color info and mouse click event
clicked = False
r=g=b=x_pos=y_pos=0
# Initialize your persistent states before the loop
target_class = None
b, g, r = 255, 255, 255# Default BLUE,GREEN,RED

# fucntion if given RED, GREEN, BULE values will give closest color to it from colors.csv.txt file
def get_color_name(R, G, B):
    minimum = 10000#greater then r+g+b(combined) 255+255+255=765<10000
    # loop through all rows in color.csv.txt and get the closest color's name R,G,B values
    for i in range(len(csv)):
        d = abs(R - int(csv.loc[i, "R"])) + abs(G - int(csv.loc[i, "G"])) + abs(B - int(csv.loc[i, "B"]))
        #print (d)
        if d <= minimum:
            minimum = d
            cname = csv.loc[i, "color_name"]
    return cname # returning color name
# function to get the color of the pixel on double click event
def draw_function(event,x,y,flags,params): 
    if event == cv2.EVENT_LBUTTONDBLCLK:
        # globally changing these variables
        global clicked,r,g,b,x_pos,y_pos
        clicked=True
        x_pos=x
        y_pos=y
        print("x and y coordinates:",x,y) #Returns x and y position while button clicking
        b,g,r=image[y,x]
        b=int(b)
        g=int(g)
        r=int(r)
        print("BGR value:",b,g,r)
# function to get the lower and upper limits of the color to detect color and its shades
def get_limits(color):
    c = np.uint8([[color]])
    hsvC = cv2.cvtColor(c, cv2.COLOR_BGR2HSV)
    hue = int(hsvC[0][0][0])
    sat = int(hsvC[0][0][1])
    val = int(hsvC[0][0][2])
    # If the clicked color has very low saturation (White, Grey, Black)
    if sat < 40 and val > 100:
        # Allow low saturation and wide value range for whites/greys
        lowerlimit = np.array([0, 0, 150], dtype=np.uint8)
        upperlimit = np.array([179, 50, 255], dtype=np.uint8)
    elif sat < 40 and val <= 100: # Blacks/Dark Greys
        lowerlimit = np.array([0, 0, 0], dtype=np.uint8)
        upperlimit = np.array([179, 50, 100], dtype=np.uint8)
    else:
        # Normal colors (Blue, Red, Orange, etc.)
        lower_hue = max(0, hue - 10)
        upper_hue = min(179, hue + 10)
        lowerlimit = np.array([lower_hue, 50, 50], dtype=np.uint8)
        upperlimit = np.array([upper_hue, 255, 255], dtype=np.uint8)

    return lowerlimit, upperlimit

output_path = "video/output.mp4"
video_path = "video/roadview.mp4"
video = cv2.VideoCapture(video_apath)#getting video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
frame_width, frame_height = 640, 468
fps = video.get(cv2.CAP_PROP_FPS)
out = cv2.VideoWriter(
    output_path, fourcc, fps if fps > 0 else 30.0, (frame_width, frame_height)
)
cv2.namedWindow('YOLO DETECTION')
cv2.setMouseCallback('YOLO DETECTION', draw_function)

# run while their are frames in the video
while video.isOpened():
  ret, bef_image = video.read()
  if not ret:# break if not frames
    break

  image = cv2.resize(bef_image, (640, 468))

  # 1. RUN YOLO DETECTION FIRST SO WE HAVE THE BOXES AVAILABLE
  results = model(image, conf=0.15, iou=0.45, imgsz=640, agnostic_nms=True)

  # 2. IF THE USER JUST DOUBLE-CLICKED, FIND WHICH BOX THEY CLICKED INSIDE
  if clicked:
    target_class = None  # Reset previous target
    for box in results[0].boxes:
      x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
      # Check if the click coordinates (x_pos, y_pos) fall inside this box
      if x1 <= x_pos <= x2 and y1 <= y_pos <= y2:
        cls_id = int(box.cls[0])
        target_class = model.names[cls_id]
        print(f"Locked onto Target Class: {target_class}")
        break
    clicked = False  # Consume the click event so it doesn't re-trigger every frame

  # 3. DISPLAY COLOR BANNER IF A COLOR HAS BEEN CHOSEN
  if target_class is not None:
    cv2.rectangle(image, (20, 20), (750, 60), (b, g, r), -1)
    text = (
        get_color_name(r, g, b)
        + f" | Target: {target_class or 'Any'}"
        + f" R={r} G={g} B={b}"
    )
    cv2.putText(image, text, (30, 50), 2, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    # for bright colors, use black text for better visibility
    if r + g + b >= 600:
      cv2.putText(image, text, (30, 50), 2, 0.6, (0, 0, 0), 2, cv2.LINE_AA)

  # Get color limits for the selected color
  lower_limit, upper_limit = get_limits((b, g, r))

  line_y_pos = 340
  cv2.line(image, (0, line_y_pos), (image.shape[1], line_y_pos), (0, 255, 0), 2)
  # variables to count the objects (on line, over line, and total objects detected)
  obj_count = 0
  obj_not_crossed = 0
  obj_over_the_line = 0
  obj_on_line_count = 0

  # 4. LOOP THROUGH BOXES AND APPLY FILTERING
  for box in results[0].boxes:
    cls_id = int(box.cls[0])
    label = model.names[cls_id]
    unique_labels.add(label)

    # If a target class was clicked, skip any boxes that don't match it
    if target_class is not None and label != target_class:
      continue

    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)
    obj_crop = image[y1:y2, x1:x2]

    if obj_crop.shape[0] > 0 and obj_crop.shape[1] > 0:
      hsv_crop = cv2.cvtColor(obj_crop, cv2.COLOR_BGR2HSV)
      mask = cv2.inRange(hsv_crop, lower_limit, upper_limit)

      matching_pixels = cv2.countNonZero(mask)
      total_pixels = obj_crop.shape[0] * obj_crop.shape[1]

      # Filter out objects that don't match the selected color profile
      if matching_pixels < (total_pixels * 0.30):
        continue

      obj_count += 1
      obj_center_y = (y1 + y2) // 2 # center of the bounding box
    # 5. DETERMINE THE POSITION OF THE OBJECT RELATIVE TO THE LINE AND DRAW ACCORDINGLY
      # left of the line
      if obj_center_y < line_y_pos - 20:# If the center of the obj is to the left of the line (with a margin of 20 pixels)
        obj_over_the_line += 1
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 255), 1)
        cv2.putText(
            image,
            f"{label}:{obj_count}",
            (x1 - 2, y1 - 7),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 0, 255),
            1,
            cv2.LINE_AA,
        )
      # on the line
      elif y1 <= line_y_pos <= y2:# If the obj's bounding box overlaps with the line (i.e., the obj is on the line)
        obj_on_line_count += 1
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 0), 1)
        cv2.putText(
            image,
            f"{label}:{obj_count}",
            (x1, y1 - 7),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 0),
            1,
            cv2.LINE_AA,
        )
      # right of the line
      elif obj_center_y > line_y_pos:# If the center of the obj is to the right of the line
        obj_not_crossed += 1
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 1)
        cv2.putText(
            image,
            f"{label}:{obj_count}",
            (x1, y1 + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

  # 6. DISPLAYING COUNTS ON SCREEN
  cv2.putText(
      image,
      f"CurrentlyonLine: {obj_on_line_count}",
      (30, 40),
      cv2.FONT_HERSHEY_SIMPLEX,
      0.6,
      (255, 255, 0),
      2,
  )
  cv2.putText(
      image,
      f"TotalSuccessfullyCrossed: {obj_over_the_line}",
      (30, 70),
      cv2.FONT_HERSHEY_SIMPLEX,
      0.6,
      (255, 0, 255),
      2,
  )
  cv2.putText(
      image,
      f"NotCrossed: {obj_not_crossed}",
      (30, 100),
      cv2.FONT_HERSHEY_SIMPLEX,
      0.6,
      (0, 255, 0),
      2,
  )
  cv2.putText(
      image,
      f"TotalDetected: {obj_count}",
      (30, 130),
      cv2.FONT_HERSHEY_SIMPLEX,
      0.6,
      (0, 0, 255),
      2,
  )
  out.write(image)
  cv2.imshow("YOLO DETECTION", image)

  # 7. KEY CONTROLS(ending video if q is pressed, reset target and color filters if 'a' is pressed)
  key = cv2.waitKey(60) & 0xFF
  if key == ord("q"):
    break
  elif key == ord("a"):  # Press 'a' to reset target and color filters
    target_class = None
    b, g, r = 255, 255, 255
    print("Filters reset.")
# 8. RELEASING VIDEO FROM MEMORY AND DESTROYING ALL WINDOWS
video.release()
out.release()
cv2.destroyAllWindows()