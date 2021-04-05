import datetime
import json


class VisitorRecord:
    def __init__(self, action, img, date):
        self.action = action
        self.img = img
        self.date = date


if __name__ == '__main__':
    vr1 = {'action': 'warning',
           'img': 'image1.jpg',
           'date': datetime.datetime(2021, 3, 3, 12, 0).strftime('%Y%m%d %H:%M')}
    vr2 = {'action': 'shot',
           'img': 'image2.jpg',
           'date': datetime.datetime(2021, 3, 3, 14, 0).strftime('%Y%m%d %H:%M')}

    records = [vr1, vr2]

    with open('data.json', 'w') as f:
        json.dump(records, f)
