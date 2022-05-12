import time

GREY = '\33[90m'
RED2 = '\33[91m'
GREEN2 = '\33[92m'
YELLOW2 = '\33[93m'
BLUE2 = '\33[94m'
VIOLET2 = '\33[95m'
BEIGE2 = '\33[96m'
WHITE2 = '\33[97m'

BLACK = '\33[30m'
RED = '\33[31m'
GREEN = '\33[32m'
YELLOW = '\33[33m'
BLUE = '\33[34m'
VIOLET = '\33[35m'
BEIGE = '\33[36m'
WHITE = '\33[37m'

COLOR_END = '\33[0m'
BOLD = '\33[1m'
ITALIC = '\33[3m'

class Logger:
    VERBOSE = 0
    DEBUG = 1
    WARN = 2
    ERROR = 3
    STATUS = 4
    OFF = 5

    def __init__(self, level=2, log_file=None):
        self.level = level
        self.log_file = log_file

    def __out(self, message, level, color):
        if level >= self.level:
            print(color + time.ctime() + " " + message + COLOR_END)
        if self.log_file:
            with open(self.log_file, "a") as fd:
                fd.write(time.ctime() + " " + message + "\n")

    def v(self, message):
        self.__out(message, Logger.VERBOSE, "")

    def d(self, message):
        self.__out(message, Logger.DEBUG, BLUE)

    def w(self, message):
        self.__out(message, Logger.WARN, YELLOW)

    def e(self, message):
        self.__out(message, Logger.ERROR, RED)

    def s(self, message):
        self.__out(message, Logger.STATUS, GREEN)
