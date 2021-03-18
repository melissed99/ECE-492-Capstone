from pygame import mixer
import time


def play_signal(duration):
    # plays alarm noise for given time in seconds
    # will stop playing after 300 seconds because only looping 100 times (3 secs each)
    mixer.init()
    warning = mixer.Sound("sound.ogg")
    warning.play(loops=100)  # sound is 3 secs long so looping for more duration/2 to give headroom
    time.sleep(duration)
    mixer.quit()


def main():
    for i in range(10):
        play_signal(1)

    # play_signal(10) -> either method works but above allows for logic in between


if __name__ == '__main__':
    main()
