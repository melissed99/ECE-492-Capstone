from flask import Flask, render_template

app = Flask(__name__)


class VisitorRecord(object):
    def __init__(self, date, image):
        self.date = date
        self.image = image


@app.route('/')
def index():
    rows = [VisitorRecord("Feb 25, 2021", "image1"), VisitorRecord("Feb 24, 2021", "image2")]
    return render_template('index.html', title='home', rows=rows)


def main():
    app.run(host='0.0.0.0', port=5000)


if __name__ == '__main__':
    main()
