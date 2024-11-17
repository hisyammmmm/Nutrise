import cv2
from mtcnn.mtcnn import MTCNN

#tes tes

detector = MTCNN()
cap = cv2.VideoCapture(0) #Mengakses kamera (index 0 berarti kamera default)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #menghubah format BGR ke RGB untuk mtcnn
    
    results = detector.detect_faces(rgb_frame) #mendeteksi wajah didalam frame dengan mtcnn

    for result in results:
        x, y, width, height = result['box']
        cv2.rectangle(frame, (x, y), (x + width, y + height), (255, 0, 0), 2) # membuat bound box pada wajah

        roi_gray = cv2.cvtColor(frame[y:y + height, x:x + width], cv2.COLOR_BGR2GRAY) # mengambil area wajah (ROI) lalu ubah ke gray scale

        smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml') # menggunakan haarcascade untuk mendeteksi senyuman
        
        smiles = smile_cascade.detectMultiScale(roi_gray, 1.8, 20) # mendeteksi senyuman dari ROI

        if len(smiles) > 0:
            cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
            cv2.putText(frame, 'Smile', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        else:
            cv2.putText(frame, ' ', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imshow('Smile Detection', frame) # menampilkan frame pada window "Smile Detection"

    if cv2.waitKey(1) & 0xFF == ord(' '): # program akan tertutup setelah menekan spasi
        break

# menghapus akses kamera dan menutup window
cap.release()
cv2.destroyAllWindows()
