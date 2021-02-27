import time
import numpy as np
import cv2
import tensorflow as tf
from flask import Flask, render_template, Response

app = Flask(__name__)


def infer_model(img, threshold=0.5):
    _t0 = time.time()
    img = img.astype(np.float32) / 255.
    img_resized = tf.image.resize(img[None, ...], (INPUT_SIZE, INPUT_SIZE))
    mask = model.predict(img_resized)[0]
    if threshold:
        mask = (mask >= threshold).astype(int)
    mask = tf.image.resize(mask, (img.shape[0], img.shape[1]))
    mask = np.concatenate([mask * 0.2, mask * 0.99, mask * 0], axis=2)
    img = np.clip((img + mask), 0, 1)
    img = (img * 255).astype(np.uint8)
    print(f'Inference time: {time.time() - _t0: .4}')
    return img, mask


saved_model_path = 'model.h5'
model = tf.keras.models.load_model(saved_model_path)
print(model.name)
INPUT_SIZE = model.input_shape[1]
print(f'{INPUT_SIZE}x{INPUT_SIZE}')


@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')


def gen(stream_url):
    """Video streaming generator function."""
    stream = cv2.VideoCapture(stream_url)
    # fps = 0  # stream.get(cv2.CAP_PROP_FPS)
    # print("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))
    prev_frame_time = 0
    time_step = 5
    thresh = .0
    mask = None
    font = cv2.FONT_HERSHEY_SIMPLEX
    t0 = time.time()

    # Read until video is completed
    while stream.isOpened():
        # Capture frame-by-frame
        ret, img = stream.read()
        if ret:
            new_frame_time = time.time()
            img = cv2.resize(img, (INPUT_SIZE, INPUT_SIZE))
            fps = int(1 / (new_frame_time - prev_frame_time))
            # img = cv2.resize(img, (0, 0), fx=0.2, fy=0.2)
            ####  #  ####
            # inference #
            #    HERE   #
            ####  #  ####
            # print(time.time() - t0)
            if time.time() - t0 >= time_step or prev_frame_time == 0:
                t0 = time.time()
                img, mask = infer_model(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), threshold=thresh)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            elif isinstance(mask, np.ndarray):
                img = img.astype(np.float32) / 255.
                img = np.clip((img + mask), 0, 1)
                img = (img * 255).astype(np.uint8)

            cv2.putText(img, text=f"FPS: {fps}", org=(16, 64), fontFace=font, fontScale=1, color=(0, 189, 255),
                        thickness=4,
                        lineType=cv2.LINE_AA)
            img = cv2.imencode('.jpg', img)[1].tobytes()
            prev_frame_time = new_frame_time
            yield b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + img + b'\r\n'
            time.sleep(0.001)
        else:
            break
    stream.release()
    print('Stream closed')
    del stream


@app.route('/video_feed_1')
def video_feed_1():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen('http://fpst-strm2.kmv.ru:9020/rtsp/6134081/c6c1f96710c09d7d8385'),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/video_feed_2')
def video_feed_2():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen('http://va-ann2.camera.rt.ru:8200/rtsp/2741058/88b4967bb8b75927cf63'),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/video_feed_3')
def video_feed_3():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen('http://va-ann4.camera.rt.ru:8400/rtsp/3362967/e8b2a0d648d2a3a7166e'),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/video_feed_4')
def video_feed_4():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen('http://fpst-strm2.kmv.ru:9020/rtsp/6134081/c6c1f96710c09d7d8385'),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
