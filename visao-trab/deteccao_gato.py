import cv2
import tensorflow as tf

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

model = tf.saved_model.load(r'C:\visao-computacional\visao-computacional\visao-trab')

LABEL_MAP = {
    1: 'pessoa',
    2: 'bicicleta',
    3: 'carro',
    4: 'motocicleta',
    5: 'avião',
    6: 'ônibus',
    7: 'trem',
    8: 'navio',
    9: 'cadeira',
    10: 'cachorro',
    11: 'gato',
    12: 'cervo',
    13: 'ovelha',
    14: 'cabra',
    15: 'cavalo',
    16: 'cachorro',
    17: 'gato',
    18: 'cachorro',
}

def run_inference_for_video(model, video_path):
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_resized = cv2.resize(frame, (320, 150))  

        frame_resized = frame_resized.astype('uint8')

        input_tensor = tf.convert_to_tensor(frame_resized)

        input_tensor = input_tensor[tf.newaxis, ...]

        output_dict = model(input_tensor)

        detection_boxes = output_dict['detection_boxes']
        detection_classes = output_dict['detection_classes']
        detection_scores = output_dict['detection_scores']

        for i in range(detection_boxes.shape[1]):
            if detection_scores[0, i] > 0.5:
                ymin, xmin, ymax, xmax = detection_boxes[0, i].numpy()
                class_id = int(detection_classes[0, i].numpy())
                score = detection_scores[0, i].numpy()

                class_name = LABEL_MAP.get(class_id, 'Desconhecido')

                (left, top, right, bottom) = (xmin * frame.shape[1], ymin * frame.shape[0],
                                              xmax * frame.shape[1], ymax * frame.shape[0])
                cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), (0, 255, 0), 2)
                cv2.putText(frame, f"{class_name} ({score:.2f})", (int(left), int(top)-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow('Detecção de Objetos', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

video_path = r'C:\visao-computacional\visao-computacional\visao-trab\gato.mp4'

run_inference_for_video(model, video_path)
