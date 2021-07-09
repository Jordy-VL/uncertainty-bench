import os
import sys
import pytest
import mongoengine
from pdb import set_trace

ARKHAM = os.path.realpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append(ARKHAM)
testpath = os.path.join(ARKHAM, "tests")
from arkham.default_config import DATAROOT


def pytest_configure():
    pytest.DATAROOT = DATAROOT  # os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


@pytest.fixture
def sample_imdb_review():
    test_text = """
    In 1989, Tim Burton created the very first Batman movie with great stars like Michael Keaton and Jack Nicholson. The Joker is definitely one of Hollywood's best villains on screen. Jack Nicholson was born for the role, with his psychotic and sick look. Michael Keaton is also great as Batman and is pretty good as Bruce Wayne. Kim Basinger is kind of annoying at times, but she's not the worst damsel in distress ever seen on screen.
    Tim Burton has a unique way of doing Batman, and I think most people can agree that it fits the characters and the story. To bad Warner Bros. got rid of him after the 2nd film.
    """
    return test_text


def pytest_addoption(parser):
    parser.addoption("--beam_width", action="store", default=3, type=int)


def pytest_generate_tests(metafunc):
    # This is called for every test. Only get/set command line arguments
    # if the argument is specified in the list of test "fixturenames".
    option_value = metafunc.config.option.beam_width
    if 'beam_width' in metafunc.fixturenames and option_value is not None:
        metafunc.parametrize("beam_width", [option_value])
