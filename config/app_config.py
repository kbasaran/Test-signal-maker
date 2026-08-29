import time
import logging
from generictools.settings import SettingsManager

APP_DEFINITIONS = {"app_name": "Test Signal Maker",
                   "version": "0.4.0",
                   "description": "Test Signal Maker - Loudspeaker test signal tool",
                   "copyright": "Copyright (C) 2026 Kerem Basaran",
                   "icon_path": "logo/icon.ico",  # relative posix path
                   "author": "Kerem Basaran",
                   "author_short": "kbasaran",
                   "email": "kbasaran@gmail.com",
                   "website": "https://github.com/kbasaran",
                   }
# uncomment for release candidate builds
APP_DEFINITIONS["version"] += "rc" + time.strftime("%y%m%d", time.localtime())

# Only the keys that generictools itself reads. The application's own settings
# are still owned by the `Settings` dataclass in main.py, which stores raw Qt
# values in the same QSettings store. Adding those field names here would let
# SettingsManager write them back as JSON strings and corrupt the dataclass on
# the next read, so keep the two sets of keys disjoint.
DEFAULTS = {
    "A_beep": 0.25,
}


def singleton_settings():
    return SettingsManager(APP_DEFINITIONS, DEFAULTS)


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    logger = logging.getLogger()

    try:
        answer = input("Type 'x' to delete all settings: ")
        if answer.lower() == 'x':
            logger.info("Deleting all settings...")
            app_settings = singleton_settings()
            app_settings.reset_all_to_defaults()
            logger.info("Settings deleted successfully.")
        else:
            logger.info("Operation cancelled by user.")
    except KeyboardInterrupt:
        exit()

else:
    logger = logging.getLogger(__name__)
