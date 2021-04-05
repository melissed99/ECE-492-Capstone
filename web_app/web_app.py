from flask import Flask, render_template, send_from_directory, request, flash, redirect, url_for
import json
import os

app = Flask(__name__, static_folder='static')

UPLOAD_FOLDER = 'uploaded_photos'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


class VisitorRecord(object):
    def __init__(self, date, action, img):
        self.date = date
        self.action = action
        self.img = img


def read_data():
    with open('data.json') as f:
        data = json.load(f)

    records = []
    for d in data:
        v = VisitorRecord(date=d['date'], action=d['action'], img=d['img'])
        records.append(v)

    return records


@app.route('/')
def index():
    return render_template('index.html', title='home')


@app.route('/records')
def display_records():
    records = read_data()
    return render_template('records.html', title='records', rows=records)


@app.route('/photos/<path:filename>')
def send_file(filename):
    return send_from_directory(app.static_folder, filename)


@app.route('/import')
def render_import():
    return render_template('import.html', title='import')


@app.route('/import', methods=['POST'])
def import_visitor_images():
    uploaded_file = request.files['file']
    if uploaded_file.filename.split('.')[1] in ALLOWED_EXTENSIONS:
        # only saves png, jpg, or jpeg files
        uploaded_file.save(os.path.join(UPLOAD_FOLDER, uploaded_file.filename))
        print('Uploaded image file: {0}'.format(uploaded_file.filename))
    else:
        print('Upload failed: {0} is not a png, jpg, or jpeg file.'.format(uploaded_file.filename))
    return redirect(url_for('render_import'))


def main():
    app.run(host='0.0.0.0', port=5000)


if __name__ == '__main__':
    main()