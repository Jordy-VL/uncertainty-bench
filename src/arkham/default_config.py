import os
import sys
import logging

logger = logging.getLogger(__name__)

CONFIG = os.path.realpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'configfile.py'))

if not os.path.isfile(CONFIG):  # defaults
    MODELROOT =  os.path.realpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'models'))
    SAVEROOT = MODELROOT
    DATAROOT = os.path.realpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'datasets')) 

else:
    sys.path.append("..")  # assume custom config is 1 level above library
    from configfile import MODELROOT, SAVEROOT, DATAROOT
