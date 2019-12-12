from flask import Flask, render_template, url_for, request, jsonify
from flask_bootstrap import Bootstrap
import PIL
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from bs4 import BeautifulSoup
from urllib.request import urlopen


app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
Bootstrap(app)


@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')


# this function loads the image into the proper dimensions, and converts to float type to be worked with
def load_img(image):
    max_dim = 512
    img = tf.io.read_file(image)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim

    new_shape = tf.cast(shape * scale, tf.int32)

    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]
    return img


# this function converts the image back to an image from a tensor
def tensorImg(tensor):
    tensor = tensor * 255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
    tensor = tensor[0]
    return PIL.Image.fromarray(tensor)


def style(image1, image2):
    content_image = load_img(image1)
    style_image = load_img(image2)
    hub_module = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')
    stylized_image = hub_module(tf.constant(content_image), tf.constant(style_image))[0]
    image = tensorImg(stylized_image)
    image.save('C:\\Sophomore Project\\static\\images\\newImage.jpg', "JPG")
    newPath = 'C:\\Sophomore Project\\static\\images\\newImage.jpg'
    return newPath


@app.route('/create/', methods=['GET', 'POST'])
def create():
    if request.method == 'POST':
        html = urlopen('http://127.0.0.1:5000/create/')
        soup = BeautifulSoup(html, 'html.parser')
        imgs = soup.find_all('img')
        for i, img in enumerate(imgs):
            if i == 0:
                img1 = img.get('src')
            if i == 1:
                img2 = img.get('src')
        return jsonify(style(str(img1), str(img2)))
    return render_template('create.html')


@app.route('/gallery/', methods=['GET', 'POST'])
def gallery():
    return render_template('gallery.html')


if __name__ == '__main__':
    app.run(debug=True)