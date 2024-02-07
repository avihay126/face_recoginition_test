# import mysql.connector
# from mysql.connector import Error
#
# def insert_image_data(image_path):
#     try:
#         # הגדרות החיבור למסד הנתונים
#         connection = mysql.connector.connect(
#             host='localhost',
#             database='ash2024',
#             user='root',
#             password='1234'
#         )
#
#         if connection.is_connected():
#             # יצירת עצם cursor
#             cursor = connection.cursor()
#
#             # קריאה לתמונה והמרתה למספר בינארי
#             with open(image_path, "rb") as image_file:
#                 binary_data = image_file.read()
#
#             # שאילתת SQL להוספת נתוני תמונה לטבלה
#             query = "INSERT INTO images (img) VALUES (%s)"
#             data = (binary_data,)
#             cursor.execute(query, data)
#
#             # אישור השינויים
#             connection.commit()
#             print("הרשומה הוספה בהצלחה.")
#
#     except Error as e:
#         print(f"שגיאה: {e}")
#
#     finally:
#         if connection.is_connected():
#             cursor.close()
#             connection.close()
#
#
# image_paths = ["C:\\Users\\DELL\\OneDrive\\שולחן העבודה\\imgTest\\IMG_0002.JPG",
#                "C:\\Users\\DELL\\OneDrive\\שולחן העבודה\\imgTest\\IMG_0003.JPG",
#                "C:\\Users\\DELL\\OneDrive\\שולחן העבודה\\imgTest\\IMG_0004.JPG",
#                "C:\\Users\\DELL\\OneDrive\\שולחן העבודה\\imgTest\\IMG_0005.JPG",
#                "C:\\Users\\DELL\\OneDrive\\שולחן העבודה\\imgTest\\IMG_0006.JPG",
#                "C:\\Users\\DELL\\OneDrive\\שולחן העבודה\\imgTest\\IMG_0007.JPG",
#                "C:\\Users\\DELL\\OneDrive\\שולחן העבודה\\imgTest\\IMG_0008.JPG",
#                "C:\\Users\\DELL\\OneDrive\\שולחן העבודה\\imgTest\\IMG_0009.JPG",
#                "C:\\Users\\DELL\\OneDrive\\שולחן העבודה\\imgTest\\IMG_0010.JPG",
#                "C:\\Users\\DELL\\OneDrive\\שולחן העבודה\\imgTest\\IMG_0011.JPG"]  # הוסף יותר נתיבים לתמונות שלך
# for path in image_paths:
#     insert_image_data(path)





# import face_recognition
# import mysql.connector
# from mysql.connector import Error
# from PIL import Image
# from io import BytesIO
# import numpy as np
# import time
#
# def get_faces_in_database(new_image_path, threshold=0.6):
#     try:
#         # חיבור למסד הנתונים
#         connection = mysql.connector.connect(
#             host='localhost',
#             database='ash2024',
#             user='root',
#             password='1234'
#         )
#
#         if connection.is_connected():
#             # יצירת עצם cursor
#             cursor = connection.cursor()
#
#             # קריאה לתמונה והמרתה למספר בינארי
#             with open(new_image_path, "rb") as image_file:
#                 binary_data = image_file.read()
#
#             # שאילתת SQL לקבלת כל התמונות בהן מופיעה פנים דומה לפנים בתמונה החדשה
#             query = "SELECT img FROM images"
#             cursor.execute(query)
#             rows = cursor.fetchall()
#             # קריאה לתמונה החדשה וזיהוי פנים בה
#             new_image = face_recognition.load_image_file(new_image_path)
#             new_face_encoding = face_recognition.face_encodings(new_image)
#
#             # השוואת פנים והדפסת תמונות שמצאו התאמה
#             for row in rows:
#                 binary_data = row[0]
#                 image = Image.open(BytesIO(binary_data))
#                 image_np = np.array(image)  # המרת PIL Image ל־NumPy array
#                 saved_face_encodings = face_recognition.face_encodings(image_np)
#                 for saved_face_encoding in saved_face_encodings:
#                     # שוואת פנים כל פנים נגד כל פנים בתמונה החדשה
#                     matches = face_recognition.compare_faces(saved_face_encoding, new_face_encoding,
#                                                              tolerance=threshold)
#
#                     # אם יש התאמה בפנים כלשהן
#                     if any(matches):
#                         print("תמונה זו דומה לפנים בתמונה מסוימת במסד הנתונים.")
#                         image.show()
#                         break
#                         # לבדוק אולי לשנות טרסהולד אבל קודם להריץ בדיבאג
#
#     except Error as e:
#         print(f"שגיאה: {e}")
#
#     finally:
#         if connection.is_connected():
#             cursor.close()
#             connection.close()
#
# # קריאה לפונקציה עבור תמונה חדשה
# new_image_path = r"C:\Users\DELL\OneDrive\שולחן העבודה\imgTest\test2\IMG_0015.JPG"  # הכנס נתיב לתמונה חדשה
# get_faces_in_database(new_image_path)






# import os
# import face_recognition
# import numpy as np
#
# def create_face_groups(image_directory):
#     # קריאה לכל התמונות בספריה
#     image_files = [f for f in os.listdir(image_directory) if f.endswith(".jpg") or f.endswith(".jpeg") or f.endswith(".JPG")]
#
#     # מיפוי של פנים למספר קבוצה
#     face_to_group_mapping = {}
#
#     # המשך רק אם יש לך תמונות
#     if image_files:
#         for image_file in image_files:
#             # קריאה לתמונה וזיהוי פנים בה
#             image_path = os.path.join(image_directory, image_file)
#             image = face_recognition.load_image_file(image_path)
#             face_locations = face_recognition.face_locations(image)
#
#             # אם יש פנים בתמונה
#             if face_locations:
#                 # אם הפנים כבר זוהו בתמונה אחרת - הוסף לאותה קבוצה
#                 for face_location in face_locations:
#                     face_encodings = face_recognition.face_encodings(image, [face_location])
#                     current_face_encoding = face_encodings[0]
#
#                     for known_face, group_id in face_to_group_mapping.items():
#                         # השוואת פנים
#                         match = face_recognition.face_distance([known_face], current_face_encoding) <= 0.6
#
#                         if any(match):
#                             face_to_group_mapping[np.frombuffer(current_face_encoding.tobytes(), dtype=np.float64)] = group_id
#                             break
#                     else:
#                         # פנים חדשות - יצירת קבוצה חדשה
#                         new_group_id = len(face_to_group_mapping) + 1
#                         face_to_group_mapping[np.frombuffer(current_face_encoding.tobytes(), dtype=np.float64)] = new_group_id
#
#     return face_to_group_mapping
#
# # קריאה לפונקציה עבור ספריית התמונות
# image_directory = r"C:\Users\DELL\OneDrive\שולחן העבודה\imgTest"
# groups = create_face_groups(image_directory)
#
# # הדפסת הקבוצות
# for face_encoding, group_id in groups.items():
#     print(f"Face encoding: {np.frombuffer(face_encoding, dtype=np.float64)}, Group ID: {group_id}")


# import cv2
# import os
# from mtcnn.mtcnn import MTCNN
#
#
# def load_images_from_folder(folder):
#     images = []
#     for filename in os.listdir(folder):
#         img_path = os.path.join(folder, filename)
#
#         img = cv2.imread(img_path)
#         if img is not None:
#             images.append(img)
#     return images
#
#
# def detect_faces(image):
#     detector = MTCNN()
#     faces = detector.detect_faces(image)
#     return faces
#
#
# def group_images_by_faces(images_folder, output_folder):
#     images = load_images_from_folder(images_folder)
#
#     grouped_faces = {}
#
#     for i, image in enumerate(images):
#         faces = detect_faces(image)
#
#         if len(faces) > 0:
#             grouped_faces[f"Group_{i + 1}"] = {"image": image, "faces": faces}
#
#             for j, face_info in enumerate(faces):
#                 x, y, w, h = face_info['box']
#                 face = image[y:y + h, x:x + w]
#
#                 # שמור כל פנים בתיקיית הפלט
#                 output_path = os.path.join(output_folder, f"Group_{i + 1}_Face_{j + 1}.jpg")
#                 cv2.imwrite(output_path, face)
#
#     return grouped_faces
#
#
# if __name__ == "__main__":
#     images_folder_path = r"C:\imgTest"
#     output_folder_path = r"C:\outputFaces"
#
#     # יצירת תיקיית הפלט אם אינה קיימת
#     if not os.path.exists(output_folder_path):
#         os.makedirs(output_folder_path)
#
#     grouped_faces = group_images_by_faces(images_folder_path, output_folder_path)
#
#     for group_name, data in grouped_faces.items():
#         print(f"Group: {group_name}, Number of Faces: {len(data['faces'])}")
#
#     print("Faces saved in the output folder.")









# import cv2
# import os
# import dlib
# def detect_faces(image_path):
#     image = cv2.imread(image_path)
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#
#     # טעינת מודל הזיהוי של Dlib
#     detector = dlib.get_frontal_face_detector()
#
#     # זיהוי פרצופים באמצעות Dlib
#     faces = detector(gray, 1)
#
#     # המרה של תוצאות הזיהוי לפורמט של OpenCV
#     faces_rect = [(face.left(), face.top(), face.width(), face.height()) for face in faces]
#
#     return faces_rect
#
# def save_faces(image_folder, output_folder):
#     for filename in os.listdir(image_folder):
#         image_path = os.path.join(image_folder, filename)
#
#         if image_path.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
#             image = cv2.imread(image_path)
#
#             if image is None:
#                 print(f"Could not load image: {filename}")
#                 continue
#
#             faces = detect_faces(image_path)
#
#             print(f"Image: {filename}, Number of Faces: {len(faces)}")
#
#             for i, (x, y, w, h) in enumerate(faces):
#                 face = image[y:y+h, x:x+w]
#                 face_filename = f"face_{i+1}_{filename}"
#                 face_path = os.path.join(output_folder, face_filename)
#                 cv2.imwrite(face_path, face)
#
# if __name__ == "__main__":
#     images_folder_path = r"C:\imgTest"
#     output_faces_folder_path = r"C:\outputFaces"
#
#     # Create the output folder if it doesn't exist
#     if not os.path.exists(output_faces_folder_path):
#         os.makedirs(output_faces_folder_path)
#
#     save_faces(images_folder_path, output_faces_folder_path)








# import cv2
# import os
# import dlib
# import face_recognition
# import numpy as np
#
#
# #constants
# IMG_FOLDER_PATH = r"C:\imgTest"
# SAVED_FOLDER_PATH = r"C:\outputFaces"
# class ImgGroup:
#
#     def __init__(self,face):
#         self.person_face = face
#         self.images_list = []
#
#     def is_same_person(self, other_face_rect):
#         # Convert the face rectangle to a NumPy array
#         other_face = np.array(other_face_rect)
#
#         # Compare faces using face_recognition library
#         return face_recognition.compare_faces([self.person_face], other_face)[0]
#
#     def save_images(self, base_folder):
#         group_folder = os.path.join(base_folder, f'Group_{id(self)}')
#         os.makedirs(group_folder, exist_ok=True)
#         for img_path in self.images_list:
#             img = cv2.imread(img_path)
#             if img is not None:
#                 cv2.imwrite(os.path.join(group_folder, os.path.basename(img_path)), img)
#
#
#
# def face_to_encoding(image, face_rect):
#     # חיתוך הפרצוף מהתמונה המלאה
#     face_image = image[face_rect[1]:face_rect[1] + face_rect[3], face_rect[0]:face_rect[0] + face_rect[2]]
#
#     # קביעת פורמט הצבע (BGR או RGB - זה תלוי בסדר פעולות קריאת התמונה)
#     rgb_face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
#
#     # יצירת מאפיינים (face encodings) לפרצוף
#     face_encodings = face_recognition.face_encodings(rgb_face_image)
#     # החזרת המאפיינים
#     return face_encodings[0] if face_encodings else None
#
#
#
#
#
# def detect_faces(image_path):
#     image = cv2.imread(image_path)
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#
#     # טעינת מודל הזיהוי של Dlib
#     detector = dlib.get_frontal_face_detector()
#
#     # זיהוי פרצופים באמצעות Dlib
#     faces = detector(gray, 1)
#
#     # המרה של תוצאות הזיהוי לפורמט של OpenCV
#     faces_rect = [(face.left(), face.top(), face.width(), face.height()) for face in faces]
#
#     return faces_rect
#
#
# def map_images(image_folder):
#     images_group =[]
#     for filename in os.listdir(image_folder):
#         image_path = os.path.join(image_folder, filename)
#         if image_path.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
#             image = cv2.imread(image_path)
#             if image is None:
#                 print(f"Could not load image: {filename}")
#                 continue
#             faces = detect_faces(image_path)
#             for face in faces:
#                 img = face_to_encoding(image, face) #צריך לבדוק למה מחזיר באחת התמונות none
#                 if img is None:
#                     continue
#                 exist = False
#                 for group in images_group:
#                     if group.is_same_person(img):
#                         group.images_list.append(image_path)
#                         exist = True
#                         break
#                 if not exist:
#                     g = ImgGroup(img)
#                     g.images_list.append(image_path)
#                     images_group.append(g)
#     save_all_images(images_group)
#
#
# def save_all_images(groups):
#     for group in groups:
#         group.save_images(SAVED_FOLDER_PATH)
#
# def main():
#     images_folder_path = IMG_FOLDER_PATH
#     map_images(images_folder_path)
#
# if __name__ == '__main__':
#     main()









import cv2
import os
import dlib
import face_recognition
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from PIL import Image

# Constants
IMG_FOLDER_PATH = r"C:\imgTest"
SAVED_FOLDER_PATH = r"C:\outputFaces"

# Load the pre-trained FaceNet model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(keep_all=True, device=device)
facenet_model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

class ImgGroup:
    def __init__(self, face):
        self.person_face = face
        self.images_list = []

    def is_same_person(self, other_face_embedding, threshold=0.6):
        # Compare face embeddings using a custom threshold
        return face_recognition.face_distance([self.person_face], other_face_embedding) <= threshold

    def save_images(self, base_folder):
        group_folder = os.path.join(base_folder, f'Group_{id(self)}')
        os.makedirs(group_folder, exist_ok=True)
        for img_path in self.images_list:
            img = cv2.imread(img_path)
            if img is not None:
                cv2.imwrite(os.path.join(group_folder, os.path.basename(img_path)), img)


def face_to_encoding(image, face_rect):
    # חיתוך הפרצוף מהתמונה המלאה
    face_image = image[face_rect[1]:face_rect[1] + face_rect[3], face_rect[0]:face_rect[0] + face_rect[2]]

    # ייבוא התמונה ל-PIL Image
    pil_image = Image.fromarray(cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB))

    # זיהוי פנים ויצירת מאפיינים פניאליים באמצעות MTCNN וFaceNet
    faces = mtcnn(pil_image)

    if faces is not None:
        with torch.no_grad():
            face_embedding = facenet_model(faces.to(device)).cpu().numpy()
            if len(face_embedding) > 0:
                return face_embedding[0]

    return None


def detect_faces(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # טעינת מודל הזיהוי של Dlib
    detector = dlib.get_frontal_face_detector()

    # זיהוי פרצופים באמצעות Dlib
    faces = detector(gray, 1)

    # המרה של תוצאות הזיהוי לפורמט של OpenCV
    faces_rect = [(face.left(), face.top(), face.width(), face.height()) for face in faces]

    return faces_rect


def map_images(image_folder):
    images_group = []
    for filename in os.listdir(image_folder):
        image_path = os.path.join(image_folder, filename)
        if image_path.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
            image = cv2.imread(image_path)
            if image is None:
                print(f"Could not load image: {filename}")
                continue
            faces = detect_faces(image_path)
            for face in faces:
                img = face_to_encoding(image, face)
                if img is None:
                    continue
                exist = False
                for group in images_group:
                    if group.is_same_person(img):
                        group.images_list.append(image_path)
                        exist = True
                        break
                if not exist:
                    g = ImgGroup(img)
                    g.images_list.append(image_path)
                    images_group.append(g)
    save_all_images(images_group)


def save_all_images(groups):
    for group in groups:
        group.save_images(SAVED_FOLDER_PATH)


def main():
    images_folder_path = IMG_FOLDER_PATH
    map_images(images_folder_path)


if __name__ == '__main__':
    main()



