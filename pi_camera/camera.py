'''
	Quick script to test the pi camera and show an example
	of how to use it.
'''

from picamera import PiCamera
from time import sleep
import os


def init_camera():
	camera = PiCamera()
	# TODO: add other camera settings
	return camera

def take_photo(camera, directory, filename):
	# takes photo

	# creates directory if it doesn't exist
	if not os.path.exists(directory):
		os.mkdir(directory)
	path = os.path.join(directory, filename)
	camera.capture(path)


def camera_preview(camera, duration):
	# starts the camera feed on the pi for duration
	camera.start_preview()
	sleep(duration)
	camera.stop_preview()


def main():
	cam = init_camera()
	camera_preview(cam, 10)
	take_photo(cam, 'photos', 'test_picture.jpg')


if __name__ == "__main__":
	main()