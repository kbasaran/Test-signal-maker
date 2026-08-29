#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
from cx_Freeze import setup, Executable
from config.app_config import APP_DEFINITIONS
from pathlib import Path
# https://cx-freeze.readthedocs.io/en/stable/setup_script.html

# Dependencies are automatically detected, but it might need fine tuning.
build_exe_options = {
    "packages": ["scipy", "numpy"],  # RecursionError in cx_Freeze if these are not provided
    "include_files": [
        (str(Path("./LICENSE")), str(Path("./LICENSE"))),
        (str(Path("./README.md")), str(Path("./README.md"))),
        (str(Path(APP_DEFINITIONS["icon_path"])), str(Path(APP_DEFINITIONS["icon_path"]))),
        ],
    "silent_level": 1,
}

bdist_msi_options = {
    # "initial_target_dir": "[ProgramFiles64Folder]" + APP_DEFINITIONS['version'],  # didn't work
    # https://cx-freeze.readthedocs.io/en/7.0.0/bdist_msi.html
    }

# base="Win32GUI" should be used only for Windows GUI app
base = "Win32GUI" if sys.platform == "win32" else None

executables=[Executable("main.py",
                        copyright=APP_DEFINITIONS["copyright"],
                        base=base,
                        shortcut_name=APP_DEFINITIONS["app_name"] + " v" + APP_DEFINITIONS["version"],
                        shortcut_dir="DesktopFolder",
                        icon=APP_DEFINITIONS["icon_path"],
                        ),
            ]

setup(
    name=APP_DEFINITIONS["app_name"],
    version=APP_DEFINITIONS["version"],
    description=APP_DEFINITIONS["description"],
    options={"build_exe": build_exe_options, "bdist_msi": bdist_msi_options},
    executables=executables,
)
