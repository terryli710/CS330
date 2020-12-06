import os
import subprocess
from shutil import copyfile

DRIVE_PATH = 'E:\GoogleDrive\Courses\CS330\HW3'
DRIVE_PYTHON_PATH = DRIVE_PATH.replace('\\', '')
if not os.path.exists(DRIVE_PYTHON_PATH):
    os.mkdir(DRIVE_PATH)

SYM_PATH = DRIVE_PATH

# download Mujoco
MJC_PATH = '{}/mujoco'.format(SYM_PATH)
if not os.path.exists(MJC_PATH):
    os.mkdir(MJC_PATH)
os.chdir(MJC_PATH)
if not os.path.exists(os.path.join(MJC_PATH, 'mujoco200')):
    subprocess.call("!wget -q https://www.roboti.us/download/mujoco200_linux.zip",
                    "!unzip -q mujoco200_linux.zip",
                    "%mv mujoco200_linux mujoco200",
                    "%rm mujoco200_linux.zip")

os.environ['LD_LIBRARY_PATH'] += ':{}/mujoco200/bin'.format(MJC_PATH)
os.environ['MUJOCO_PY_MUJOCO_PATH'] = '{}/mujoco200'.format(MJC_PATH)
os.environ['MUJOCO_PY_MJKEY_PATH'] = '{}/mjkey.txt'.format(MJC_PATH)

# setup virtual display
from pyvirtualdisplay import Display
display = Display(visible=0, size=(1400, 900))
display.start()