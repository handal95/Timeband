import os
import enum
import pytz
import structlog
import datetime
import logging.config

KST = pytz.timezone("Asia/Seoul")
LOG_FORMAT = structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S", utc=False)
pre_chain = [structlog.stdlib.add_log_level, LOG_FORMAT]

today = datetime.datetime.today()


class VerbosityEnum(enum.Enum):
    TRACE = 0
    DEBUG = 1
    INFO = 2
    ERROR = 3
    WARN = 4
    OFF = 5

class Logger:
    # Logger verbosity level
    def __init__(self, logfile: str, verbosity: int = VerbosityEnum.DEBUG):
        # Set log file path
        self.logfile = logfile
        self.verbosity = verbosity

        self.setup_logger()
        self.logger = logging.getLogger(__name__)
        self.logger.propagate = True
        
    def setup_logger(self):
        _LOG_DICT = {
            "version": 1,
            "disable_existing_loggers": True,
            "formatters": {
                "plain": {
                    "()": structlog.stdlib.ProcessorFormatter,
                    "processor": structlog.dev.ConsoleRenderer(colors=False),
                    "foreign_pre_chain": pre_chain,
                },
                "colored": {
                    "()": structlog.stdlib.ProcessorFormatter,
                    "processor": structlog.dev.ConsoleRenderer(colors=True),
                    "foreign_pre_chain": pre_chain,
                },
            },
            "handlers": {
                "default": {
                    "level": "DEBUG",
                    "class": "logging.StreamHandler",
                    "formatter": "colored"
                },
            },
            "loggers": {
                "": {
                    "handlers": ["default"],
                    "level": "DEBUG",
                }
            },
        }
        
        if hasattr(self, "filepath"):
            _LOG_DICT["loggers"][""]["handlers"].append("file")
            _LOG_DICT["handlers"]["file"] = {
                "level": "INFO",
                "class": "logging.handlers.RotatingFileHandler",
                "filename": f"{self.logfile}.log",
                "formatter": "plain",
                "maxBytes": 1024,
                "backupCount": 3
            }
        
        logging.config.dictConfig(_LOG_DICT)

    def debug(self, message: str):
        return self.logger.debug(message)

    def info(self, message: str):
        return self.logger.info(message)

    def warn(self, message: str):
        return self.logger.warning(message)
