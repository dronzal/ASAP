from subprocess import Popen, PIPE
import getpass
import sys

def load_driver(name="ASAP_CAM", pwd=None):
    if sys.platform.startswith('linux'):
        load_linux_driver(name, pwd)

    elif sys.platform.startswith('windows'):
        load_linux_driver(name, pwd)

def load_linux_driver(name, pwd):
    # check if pwd is setted
    if pwd:
        # init task string
        load_linux_driver = f'sudo -S modprobe v4l2loopback devices=1 card_label={name}'

        # subprocess needs a list
        load_linux_driver = load_linux_driver.split(" ")

        # open a pipe with the subprocess
        proc = Popen(load_linux_driver, stdin=PIPE)

        # communicatie with the pipe
        proc.communicate(input=pwd.encode('UTF-8'))

def load_windows_driver(name, pwd):
    return True


