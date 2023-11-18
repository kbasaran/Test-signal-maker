#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
from cx_Freeze import setup, Executable
from main import app_definitions
from pathlib import Path
# https://cx-freeze.readthedocs.io/en/stable/setup_script.html

# Dependencies are automatically detected, but it might need fine tuning.
build_exe_options = {
    "packages": ["numpy"],
    "include_files": [
        (str(Path("./LICENSE")), str(Path("./LICENSE"))),
        (str(Path("./README.md")), str(Path("./README.md"))),
	(app_definitions["icon_path"], app_definitions["icon_path"]),
        ],
    "silent_level": 1,
}

bdist_msi_options = {
                  }

# base="Win32GUI" should be used only for Windows GUI app
base = "Win32GUI" if sys.platform == "win32" else None

executables=[Executable("main.py",
                        copyright=app_definitions["copyright"],
                        base=base,
                        shortcut_name=app_definitions["app_name"] + " v" + app_definitions["version"],
                        shortcut_dir="DesktopFolder",
                        icon=app_definitions["icon_path"],
                        ),
            ]

setup(
    name=app_definitions["app_name"],
    version=app_definitions["version"],
    description=app_definitions["description"],
    options={"build_exe": build_exe_options, "bdist_msi": bdist_msi_options},
    executables=executables,
)
