import time
import cv2
from flask import Flask, render_template, Response

app = Flask(__name__)


@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')


def gen(stream_url):
    """Video streaming generator function."""
    stream = cv2.VideoCapture(stream_url)
    fps = 0  # stream.get(cv2.CAP_PROP_FPS)
    print("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))
    prev_frame_time = 0
    font = cv2.FONT_HERSHEY_SIMPLEX
    # Read until video is completed
    while stream.isOpened():
        # Capture frame-by-frame
        ret, img = stream.read()
        if ret:
            new_frame_time = time.time()
            img = cv2.resize(img, (512, 512))
            fps = int(1 / (new_frame_time - prev_frame_time))
            # img = cv2.resize(img, (0, 0), fx=0.2, fy=0.2)
            ####  #  ####
            # inference #
            #    HERE   #
            ####  #  ####
            cv2.putText(img, text=str(fps), org=(16, 80), fontFace=font, fontScale=1, color=(0, 189, 255), thickness=4,
                        lineType=cv2.LINE_AA)
            frame = cv2.imencode('.jpg', img)[1].tobytes()
            prev_frame_time = new_frame_time
            yield b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n'
            time.sleep(0.005)
        else:
            break


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
