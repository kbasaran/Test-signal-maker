# Test Signal Maker - Loudspeaker testing tool
# Copyright (C) 2023 - Kerem Basaran
# https://github.com/kbasaran
__email__ = "kbasaran@gmail.com"

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

# from datetime import date
# today = date.today()
from pathlib import Path

app_definitions = {"app_name": "Test Signal Maker",
                   "version": "0.2.0",
                   # "version": "Test build " + today.strftime("%Y.%m.%d"),
                   "description": "Test Signal Maker - Loudspeaker test signal tool",
                   "copyright": "Copyright (C) 2023 Kerem Basaran",
                   "icon_path": str(Path("./logo/icon.ico")),
                   "author": "Kerem Basaran",
                   "author_short": "kbasaran",
                   "email": "kbasaran@gmail.com",
                   "website": "https://github.com/kbasaran",
                   }

import sys
import os
import time as pyt_time

# https://doc.qt.io/qtforpython/
from PySide6 import QtWidgets as qtw
from PySide6 import QtCore as qtc
from PySide6 import QtGui as qtg

import sounddevice as sd  # https://python-sounddevice.readthedocs.io
import numpy as np
import acoustics as ac  # https://github.com/timmahrt/pyAcoustics
import soundfile as sf  # https://python-soundfile.readthedocs.io/
from scipy import signal
import copy
from datetime import datetime

import matplotlib.pyplot as plt  # http://matplotlib.org/
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

from generictools.signal_tools import TestSignal, make_fade_window_n
from dataclasses import dataclass, fields

import logging

home_folder = os.path.expanduser("~")
logging.basicConfig(level=logging.INFO,
                    filename=os.path.join(home_folder, 'tsm.log'),
#                    encoding='utf-8',
                    format='%(asctime)s %(levelname)s - %(funcName)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    )



class FileImportDialog(qtw.QDialog):

    def __init__(self, main_win_signals):
        super().__init__()
        # self.setModal(True)
        self.setWindowTitle("Import file")
        self.setMinimumSize(120, 100)

        self.choose_channel_label = qtw.QLabel("Channel to use")
        self.choose_channel_combo = qtw.QComboBox(enabled=False)
        self.choose_channel_combo.addItem("Channel 1", None)
        self.choose_channel_combo.setCurrentIndex(0)

        self.sample_rate_label = qtw.QLabel("Sample rate")
        self.sample_rate_combo = qtw.QComboBox(enabled=False)

        self.buttonBox = qtw.QDialogButtonBox(qtw.QDialogButtonBox.Ok | qtw.QDialogButtonBox.Cancel)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)

        # Layout
        self.layout = qtw.QVBoxLayout()
        self.form_layout = qtw.QFormLayout()
        self.setLayout(self.layout)

        self.layout.addWidget(self.choose_file_button)
        self.layout.addLayout(self.form_layout)
        self.layout.addWidget(self.buttonBox, alignment=qtc.Qt.AlignHCenter)
        self.form_layout.addRow("Channel to use:", self.choose_channel_combo)
        self.form_layout.addRow("Sample rate", self.sample_rate_combo)


# Popup window for warnings
class PopupError():
    # https://www.techwithtim.net/tutorials/pyqt5-tutorial/messageboxes/
    def __init__(self, text, informative_text=None, post_action=None, title="Error"):
        msg = qtw.QMessageBox()
        msg.setWindowTitle(title)
        msg.setText(text)
        # msg.setMinimumWidth(200)  # doesn't work
        msg.setInformativeText(informative_text)
        msg.setIcon(qtw.QMessageBox.Warning)
        msg.setStandardButtons(qtw.QMessageBox.Ok)

        def ok_button_pressed():
            if post_action:
                post_action()

        msg.buttonClicked.connect(ok_button_pressed)
        msg.exec()


class SysGainAndLevelsPopup(qtw.QDialog):
    global settings

    user_changed_sys_params_signal = qtc.Signal()
    # channel_count_changed = qtc.Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setWindowTitle("System parameters")
        self.setMinimumWidth(400)
        self.setMinimumHeight(200)

        # Form for gains
        sys_gain_form_layout = qtw.QFormLayout()

        preferred_device_name = settings.preferred_device

        preferred_device_widget = qtw.QComboBox()
        for device in sd.query_devices():
            hostapi_name = sd.query_hostapis(index=device['hostapi'])['name']
            if device["max_output_channels"] > 0 and "WDM" not in hostapi_name and "MME" not in hostapi_name:
                device_name = device['name']
                data_name = hostapi_name + " - " + device_name
                user_friendly_name = f"{data_name} - {device['max_output_channels']} channels"
                preferred_device_widget.addItem(user_friendly_name, data_name)  # data is the pure name from sounddevice. sometimes duplicate.
        preferred_device_index = preferred_device_widget.findData(preferred_device_name)  # -1 needs not found, and empty selection
        if preferred_device_index == -1:
            preferred_device_index = sd.default.device[1]
        preferred_device_widget.setCurrentIndex(preferred_device_index)  # does this raise an error if that device name is not in the combobox?
        sys_gain_form_layout.addRow("Preferred device", preferred_device_widget)

        number_of_channels_widget = qtw.QSpinBox(Minimum=2,
                                                 Maximum=int(settings.max_channel_count),
                                                 ToolTip="Number of amplifier channels that should become available.",
                                                 Value=int(settings.channel_count),
                                                 )

        sys_gain_form_layout.addRow("Active channels", number_of_channels_widget)

        channel_gain_widgets = {}
        for cn in range(1, int(settings.max_channel_count) + 1):
            channel_gain_widgets[cn] = qtw.QDoubleSpinBox(Minimum=-100,
                                                         Maximum=200,
                                                         SingleStep=0.1,
                                                         Value=float(settings.system_gains[cn-1]),
                                                         ToolTip="\n".join(["in dB, Volts per full scale.",
                                                                            "e.g. setting 26 means a full scale sine wave is creating 20V peaks at amplifier output.",
                                                                            "1 * 10^(26/20) = 20V"])
                                                         )
            sys_gain_form_layout.addRow(f"System gain for Ch. {cn}", channel_gain_widgets[cn])

        # Peak amp voltage
        amp_peak_capability_widget = qtw.QDoubleSpinBox(Minimum=0.0001,
                                                        Maximum=999,
                                                        Value=settings.amp_peak,
                                                        )

        # Sweep sample rate
        sweep_sample_rate = qtw.QComboBox()
        for val in [44100, 48000, 96000]:
            sweep_sample_rate.addItem(str(val), val)

        # check which value is stored in settings
        current_val = settings.sweep_sample_rate
        current_val_idx = sweep_sample_rate.findData(current_val)

        if current_val_idx == -1:  # if the sapmle rate is not available in current list
            sweep_sample_rate.setCurrentIndex(0)
        else:
            sweep_sample_rate.setCurrentIndex(current_val_idx)

        # Stream latency
        stream_latency = qtw.QComboBox()
        stream_latency.addItem("Sound device default: High", "high")
        stream_latency.addItem("Sound device default: Low", "low")
        stream_latency.addItem("User value: Safe - 50ms", 0.05)
        stream_latency.addItem("User value: Very safe - 100ms", 0.1)

        current_val = settings.stream_latency
        current_val_idx = stream_latency.findData(current_val)

        if current_val_idx == -1:
            stream_latency.setCurrentIndex(0)
        else:
            stream_latency.setCurrentIndex(current_val_idx)

        # Rest of the layouts
        sys_gain_form_layout.addWidget(qtw.QFrame(FrameShape=qtw.QFrame.HLine, FrameShadow=qtw.QFrame.Sunken))
        sys_gain_form_layout.addRow("Amplifier peak capability (V)", amp_peak_capability_widget)

        sys_gain_form_layout.addWidget(qtw.QFrame(FrameShape=qtw.QFrame.HLine, FrameShadow=qtw.QFrame.Sunken))
        sys_gain_form_layout.addRow("Sweep sample rate", sweep_sample_rate)
        sys_gain_form_layout.addRow("Stream latency", stream_latency)

        # Pushbutton
        save_sys_gain_settings = qtw.QPushButton("Save and close")
        save_sys_gain_settings.setStyleSheet("padding: 7px;")

        # Make layout
        gain_and_levels_window_layout = qtw.QVBoxLayout()
        self.setLayout(gain_and_levels_window_layout)

        gain_and_levels_window_layout.addLayout(sys_gain_form_layout)
        gain_and_levels_window_layout.addWidget(save_sys_gain_settings, alignment=qtc.Qt.AlignHCenter)

        @qtc.Slot(int)
        def disable_inactive_channels_widgets(channel_count):
            for i in range(1, int(settings.max_channel_count) + 1):
                channel_gain_widgets[i].setEnabled(i <= channel_count)
        number_of_channels_widget.valueChanged.connect(disable_inactive_channels_widgets)

        # Run once at start_up:
        disable_inactive_channels_widgets(number_of_channels_widget.value())

        def save_and_close():
            settings.update_attr("preferred_device", preferred_device_widget.currentData())
            system_gains = list(settings.system_gains)
            for cn in channel_gain_widgets.keys():
                system_gains[cn-1] = channel_gain_widgets[cn].value()
            settings.update_attr("system_gains", tuple(system_gains))
            settings.update_attr("amp_peak", amp_peak_capability_widget.value())
            settings.update_attr("channel_count", number_of_channels_widget.value())
            settings.update_attr("max_channel_count", settings.max_channel_count)  # so this stays fixed, no user option to change it yet
            settings.update_attr("sweep_sample_rate", sweep_sample_rate.currentData())
            settings.update_attr("stream_latency", stream_latency.currentData())

            self.user_changed_sys_params_signal.emit()
            self.done(0)
        save_sys_gain_settings.clicked.connect(save_and_close)
        
        def update_max_channel_count(device_index):
            device = sd.query_devices()[device_index]
            number_of_channels_widget.setMaximum(max(device["max_output_channels"],
                                                     int(settings.max_channel_count),
                                                     )
                                                 )
        preferred_device_widget.currentIndexChanged.connect(update_max_channel_count)


class Generator(qtc.QObject):

    """
    Signal generator object.

    """
    # signals need to be class variables as you see. don't know why. PyQt thing.
    signal_ready = qtc.Signal(TestSignal)
    file_import_success = qtc.Signal(TestSignal)
    busy = qtc.Signal(str)
    exception = qtc.Signal(Exception)

    @qtc.Slot(str)
    def import_file(self, import_file_path):
        logging.debug(f'Importing file...{import_file_path}')
        self.busy.emit("Importing file...")
        try:
            self.imported_signal = TestSignal("Imported",
                                              import_file_path=import_file_path,
                                              import_channel="downmix_all",
                                              )
            self.file_import_success.emit(self.imported_signal)
            logging.debug("Signal imported and params published.")

        except Exception as e:
            # Pop-up
            logging.error("Import failed" + str(e))
            self.exception.emit(e)

    @qtc.Slot()
    def clear_imported_file(self):
        self.imported_signal = None

    @qtc.Slot(str, dict)
    def process_imported_file(self, sig_type, kwargs):
        self.busy.emit("Generating from import...")
        try:
            if not hasattr(self, "imported_signal") or not self.imported_signal:
                raise KeyError("No file imported to process.")
            generated_signal = copy.deepcopy(self.imported_signal)
            generated_signal.reuse_existing(**kwargs)
            self.signal_ready.emit(generated_signal)
            logging.debug("Imported signal has been processed and published.")

        except Exception as e:
            # Pop-up
            logging.error(str(e))
            self.exception.emit(e)

    @qtc.Slot(str, dict)
    def generate_ugs(self, sig_type, kwargs):
        logging.debug(f'Generate ugs "{sig_type}" initiated')
        self.busy.emit(f"Generating {sig_type.lower()}...")
        try:
            generated_signal = TestSignal(sig_type, **kwargs)
            self.signal_ready.emit(generated_signal)
            logging.info(f"Signal with type {sig_type} generated.")

        except Exception as e:
            # Pop-up
            logging.error(str(e))
            self.exception.emit(e)


class LogView(qtw.QDialog):
    def __init__(self, input_dict):
        super().__init__()

        fig = plt.Figure()
        self.canvas = FigureCanvas(fig)
        self.toolbar = NavigationToolbar(self.canvas, self)

        lay = qtw.QVBoxLayout(self)
        lay.addWidget(self.toolbar)
        lay.addWidget(self.canvas)
        lay.setContentsMargins(0, 0, 0, 0)
        self.ax = fig.add_subplot(111)
        fig.tight_layout()

        self.setLayout(lay)

        self.update_plot(input_dict)

    @qtc.Slot(list)
    def update_plot(self, input_dict):
        self.ax.cla()
        self.ax.plot(input_dict["time_sig"], ".-")
        self.ax.plot(np.array(input_dict["fade_out_window"]) * np.max(input_dict["time_sig"]), "-m")
        self.ax.plot(np.array(input_dict["fade_in_window"]) * np.max(input_dict["time_sig"]), "-g")
        self.ax.grid(which='minor', axis='x')
        self.ax.grid(which='major', axis='y')
        # self.ax.legend()

        self.canvas.draw()

    def clear_plot(self):
        self.update_plot(None)


class Player(qtc.QObject):
    global settings

    play_stopped = qtc.Signal(str)
    play_started = qtc.Signal(str)
    sweep_generated = qtc.Signal(float, float)
    sweep_generator_stopped = qtc.Signal(str)
    signal_sound_devices_polled = qtc.Signal(str)
    signal_exception = qtc.Signal(str)
    publish_log = qtc.Signal(dict)
    impossible_voltage_request = qtc.Signal(str)
    log_through_thread = qtc.Signal(str)

    # which methods should have exception handling in them?

    def __init__(self):
        super().__init__()

        # default setting for sd
        sd.default.prime_output_buffers_using_stream_callback = True

        # pre-assign state variables for sweep generator
        self._theta_last = np.nan
        self._omega_last = np.nan
        self.play_pos = None
        self.is_play_in_loop = False
        self.reset_fade_out()
        self.output_log = {"time_sig": [],
                           "fade_out_window": np.array([]),
                           "fade_in_window": [],
                           }
        self.log_output_signal = False  # only logs channel 1 currently

        # define the sound device based on settings and availability
        self.find_right_sound_device()
        
        # Inititate attributes
        self._sweep_voltage = 0
        self._sweep_channel = 0

    def reset_fade_out(self):
        self.fade_out_frames = {"remaining": np.nan,
                                "total": np.nan,
                                "stop_after": None,
                                }

    def announce_callback_is_finished(self):
        self.play_pos = None
        self.reset_fade_out()
        self.user_req_omega, self.user_req_alpha = np.nan, np.nan
        self.sweep_generator_stopped.emit("Stopped")
        self.play_stopped.emit("Stopped.")
        self._theta_last = np.nan
        self._omega_last = np.nan
        self._bring_wave_states_to_zero(self.stream.channels)
        self.sweep_generated.emit(np.nan, np.nan)
        self.fade_out_frames = {"remaining": np.nan, "total": np.nan}
        if self.log_output_signal:
            self.publish_log.emit(self.output_log)
            self.output_log = {"time_sig": [],
                               "fade_out_window": [],
                               "fade_in_window": [],
                               }
        logging.info("Audio stream stopped.")
        return None

    def find_right_sound_device(self):
        preferred_device_name = settings.preferred_device
        device_name_to_index = {}
        for device in sd.query_devices():
            hostapi_name = sd.query_hostapis(index=device['hostapi'])['name']
            device_name = device['name']
            data_name = hostapi_name + " - " + device_name
            device_name_to_index[data_name] = device["index"]

        self.play_device_idx = device_name_to_index.get(preferred_device_name, sd.default.device[1])
        # 0 is the recording device, 1 is playback
        # sd.default.device returns (int, int)

    @qtc.Slot()
    def poll_sound_devices(self):
        try:
            # this is invoked regularly to update the current sound device in use
            # without this, when default device is changed in operating system, there is no detection
            # https://github.com/spatialaudio/python-sounddevice/issues/337
            # if hasattr(self, "stream") and not self.stream.active:  # if stream is not active
                # sd._terminate()
                # sd._initialize()
            self.find_right_sound_device()

            play_device_info = sd.query_devices(self.play_device_idx)
            # this doesn't update when default sound device is changed in operating system :(
            # thus the trick above
            play_device_summary = f"""Device name: {play_device_info['name']}
--Host API: {sd.query_hostapis()[play_device_info['hostapi']]["name"]}
--Max. output channels: {play_device_info['max_output_channels']}
--Default samplerate: {int(play_device_info['default_samplerate'])}
--Default data type: {sd.default.dtype[1]}
"""
            if hasattr(self, "stream"):
                play_device_summary += f"--Reported latency: {self.stream.latency * 1000:.3g}ms"

        except Exception as e:
            play_device_summary = f"Exception while detecting sound devices.\n{e}"

        self.signal_sound_devices_polled.emit(play_device_summary)

    def calculate_digital_signal_rms(self, requested_voltages: dict, signal_CF: float) -> list:
        """
        Calculates the digital signal rms that is necessary to get the correct voltage output.
        Also checks if this signal level might cause clipping.
        """
        if not isinstance(requested_voltages, dict) or not isinstance(signal_CF, (float, int)):
            self.signal_exception.emit("Incorrect data type received for voltage to digital signal RMS conversion.")
            raise sd.CallbackAbort

        amp_peak_voltage_capability = settings.amp_peak
        channels = requested_voltages.keys()
        rms_for_digital_signals = {cn: requested_voltages[cn] / 10**(settings.system_gains[cn-1] / 20) for cn in channels}
        peak_for_digital_signals = {cn: rms * signal_CF for cn, rms in rms_for_digital_signals.items()}

        peak_voltages = {cn: requested_voltages[cn] * signal_CF for cn in channels}

        if max(peak_for_digital_signals.values()) > 1:
            error_text = "Current settings will cause digital clipping at sound card output."
            informative_text = ("Increase amplifier gain or reduce target RMS voltage and/or crest factor."
                                + "\nMake sure system gain is entered correctly."
                                )
            self.stop_play()
            self.signal_exception.emit(error_text + "\n" + informative_text)
            self.impossible_voltage_request.emit(error_text)

        elif max([val for val in peak_voltages.values()]) > amp_peak_voltage_capability:  # val is in abs. why?
            error_text = f"Required peaks exceed amplifier peak voltage capability of {self._sys_params['amp_peak']} V."
            informative_text = ("Reduce target RMS voltage and/or crest factor."
                                + "\nMake sure system gain and amplifier peak voltage capability is entered correctly."
                                )
            self.stop_play()
            self.signal_exception.emit(error_text + "\n" + informative_text)
            self.impossible_voltage_request.emit(error_text)

        else:
            return rms_for_digital_signals

    def _initiate_stream(self, stream_settings):
        """Prepare the stream object with provided settings"""
        self.stop_play()

        # Check if anything is new - (doesn't work properly due to .latency receiving str but returning float)
        new_settings = {}
        running_stream = None if not hasattr(self, "stream") else self.stream
        for setting, value in stream_settings.items():
            if getattr(running_stream, setting, None) != value:  # this returns None if there is no such key
                new_settings.update({setting: value})

        # Logging
        if new_settings:
            logging.debug(f"New stream settings: {new_settings}")

        # If anything is new or there was no stream in the first place
        if new_settings:
            if hasattr(self, "stream"):
                while self.stream.active: # maybe it is still doing callbacks, so wait a bit
                    pass
                self.stream.close()

            self._bring_wave_states_to_zero(stream_settings["channels"])

            self.stream = sd.OutputStream(callback=self.callback,
                                          device=self.play_device_idx,
                                          finished_callback=self.announce_callback_is_finished,
                                          **stream_settings,
                                          )
            self.fade_window_size = int(self.stream.samplerate // 10)
            self.ugs_play_elapsed_time = -1.

    def _bring_wave_states_to_zero(self, channel_count):
        self._omega_last = 0.
        self._theta_last = 0.
        self._sweep_level_last = {channel: 0. for channel in range(1, channel_count + 1)}


    @qtc.Slot()
    def is_active(self):
        return hasattr(self, "stream") and self.stream.active

    def calculate_quiet(self, t_array, theta_start, omega_start, omega_end):
        """
        Calculate a quiet section
        Returns a tuple with,
        one channel array of theta, last value of theta, last value of omega
        """
        mono_signal_chunk = np.zeros(len(t_array))
        theta_last = 0
        omega_last = 0
        return mono_signal_chunk, theta_last, omega_last

    def calculate_lin_sweep(self, t_array, theta_start, omega_start, omega_end):
        """
        Calculate a linear sine sweep
        Returns a tuple with,
        one channel array of theta, last value of theta, last value of omega
        """
        T = t_array[-1]
        alpha = (omega_end - omega_start) / T
        theta_array = (theta_start + omega_start * t_array + alpha * t_array**2 / 2) % (2 * np.pi)
        mono_signal_chunk = np.sin(theta_array)
        theta_last = theta_array[-1]
        omega_last = omega_start + alpha * T
        return mono_signal_chunk, theta_last, omega_last

    def calculate_exp_sweep(self, t_array, theta_start, omega_start, omega_end):
        """
        Calculate an exponential sine sweep
        Returns a tuple with,
        one channel array of theta, last value of theta, last value of omega
        """
        T = t_array[-1]
        n = (omega_end / omega_start)**(1 / T)
        k = omega_start / np.log(n)
        theta_array = (theta_start + k * (np.exp(t_array * np.log(n)) - 1)) % (2 * np.pi)
        mono_signal_chunk = np.sin(theta_array)
        theta_last = theta_array[-1]
        omega_last = omega_end
        return mono_signal_chunk, theta_last, omega_last

    def calculate_exp_sweep_with_acceleration():
        """
        Calculate an exponential sine sweep with user defined acceleration.
        Returns a tuple with,
        one channel array of theta, last value of theta, last value of omega
        """
        pass

    def callback_for_ugs(self, frames):
        # We are doing a callback for streaming an already generated signal ------------------
        do_callback_stop = False
        try:
            # Try to fill the soundcard buffer within this loop
            empty_frames = int(frames)
            mono_signal_chunk = np.empty(frames)

            while empty_frames > 0:
                logging.debug("---Fill cycle---")
                logging.debug(f"Play pos: {self.play_pos}")
                len_user_signal = len(self.user_gen_signal.time_sig)
                remaining_in_user_signal = len_user_signal - self.play_pos

                if remaining_in_user_signal > 0:
                    number_of_samples_to_write = min(remaining_in_user_signal, empty_frames)
                    part_mono_signal_chunk = self.user_gen_signal.time_sig[self.play_pos:self.play_pos + number_of_samples_to_write]
                else:  # fill it all with empty
                    number_of_samples_to_write = empty_frames
                    part_mono_signal_chunk = np.zeros(empty_frames)

                # trigger fade_out because end of the signal is coming
                if (remaining_in_user_signal <= self.fade_window_size) and np.isnan(self.fade_out_frames["remaining"]):
                    self.fade_out_frames = {"remaining": remaining_in_user_signal,
                                            "total": remaining_in_user_signal,
                                            "stop_after": False,
                                            }

                logging.debug(f"Fade out frames: {self.fade_out_frames}")

                # reached end of fade-out and not gonna loop, so stop calling back
                if (self.fade_out_frames["remaining"] <= empty_frames) and (not self.is_play_in_loop or self.fade_out_frames["stop_after"]):
                    do_callback_stop = True

                # Apply fade-out
                if self.fade_out_frames["remaining"] > 0:
                    fade_start_end_idx = (self.fade_out_frames["remaining"] - self.fade_out_frames["total"],
                                          self.fade_out_frames["remaining"],
                                          )
                    fade_out_window = make_fade_window_n(1,
                                                         0,
                                                         number_of_samples_to_write,
                                                         fade_start_end_idx,
                                                         )

                    part_mono_signal_chunk = part_mono_signal_chunk * fade_out_window

                    if self.log_output_signal:
                        self.output_log["fade_out_window"] = np.concatenate([self.output_log["fade_out_window"], fade_out_window])

                    self.fade_out_frames["remaining"] -= number_of_samples_to_write

                else:
                    if self.log_output_signal:
                        self.output_log["fade_out_window"] = np.concatenate([self.output_log["fade_out_window"], np.ones(number_of_samples_to_write) * np.nan])

                # note: the player tab is not disabled during playing a signal

                # Apply fade-in
                if self.play_pos < self.fade_window_size:
                    fade_start_end_idx = (-self.play_pos,
                                          self.fade_window_size - self.play_pos,
                                          )
                    fade_in_window = make_fade_window_n(0,
                                                        1,
                                                        number_of_samples_to_write,
                                                        fade_start_end_idx,
                                                        )
                    part_mono_signal_chunk = part_mono_signal_chunk * fade_in_window

                if self.log_output_signal:
                    window_to_write = list(fade_in_window if self.play_pos < self.fade_window_size else np.ones(number_of_samples_to_write) * np.nan)
                    self.output_log["fade_in_window"].extend(window_to_write)

                # add the data from this while loop to the temporary signal block
                start_position = frames - empty_frames                    
                mono_signal_chunk[start_position:(start_position + number_of_samples_to_write)] = part_mono_signal_chunk

                empty_frames -= number_of_samples_to_write

                self.play_pos += number_of_samples_to_write
                if self.is_play_in_loop:
                    self.play_pos = self.play_pos % len_user_signal

            # Reset the fade-out counters when fade-out is over
            if (self.fade_out_frames["remaining"] <= 0):
                self.reset_fade_out()

            # Make a table with correct rms signal levels
            initial_rms = self.user_gen_signal.RMS
            ugs_play_rms_levels = [None] * self.stream.channels
            for channel in range(1, self.stream.channels + 1):
                ugs_play_rms_levels[channel - 1] = self._ugs_play_signal_rms[channel]
            if self.log_output_signal:
                logging.debug(f"User generated signal play levels: {ugs_play_rms_levels:.4f}")

            if self.ugs_play_elapsed_time == -1.:
                self.log_through_thread.emit(f"Started with: {self._ugs_play_voltages}Vrms")
            self.ugs_play_elapsed_time += self.stream.latency
            if self.ugs_play_elapsed_time > 60 * 60 * 6:  # 6 is a correction factor. latency value from sound card is incorrect
                self.log_through_thread.emit(f"Ongoing with: {self._ugs_play_voltages}Vrms")
                self.ugs_play_elapsed_time = 0.
            return mono_signal_chunk, initial_rms, ugs_play_rms_levels, do_callback_stop

        except Exception as e:
            logging.critical(
                f"Failed to add {frames} frames during usg callback." +
                f"\nPosition: {self.play_pos}/{len(self.user_gen_signal.time_sig)}. Error: {str(e)}")
            raise sd.CallbackAbort  # why?

    def callback_for_sweep(self, frames):
        # We are doing a frequency generator callback --------------------
        do_callback_stop = False

        try:
            target_omega, alpha = self.user_req_omega, self.user_req_alpha

            # Our time array for this callback.
            # 0 represents latest value therefore starting from 1
            t_array = np.arange(1, frames + 1) / self.stream.samplerate

            # if user requests acceleration and not a target omega
            # this will translate alpha and change target_omega from nan to a value
            if np.isnan(target_omega) and not np.isnan(alpha):
                target_omega = min(self.stream.samplerate / 3,
                                   max(0,
                                       self._omega_last * 2**(alpha * frames),
                                       )
                                   )

            # Exponential sweep is necessary
            if target_omega > 0 and self._omega_last > 0 and (target_omega != self._omega_last):
                logging.debug("Callback case exponential")
                mono_signal_chunk, self._theta_last, self._omega_last =\
                    self.calculate_exp_sweep(t_array,
                                             self._theta_last,
                                             self._omega_last,
                                             target_omega,
                                             )
                # output should be faded out to zero if target is 0Hz.

            # Need to be quiet
            elif target_omega == 0 and self._omega_last == 0:
                logging.debug("Callback case zero output")
                # Otherwise it clicks.
                mono_signal_chunk = np.zeros(frames)

            # Linear sweep is necessary
            else:
                logging.debug("Callback case linear")
                mono_signal_chunk, self._theta_last, self._omega_last =\
                    self.calculate_lin_sweep(t_array,
                                             self._theta_last,
                                             self._omega_last,
                                             target_omega,
                                             )

            # There was a omega=0 case. Reset the theta and omega values.
            if target_omega == 0:
                self._bring_wave_states_to_zero(self.stream.channels)

            # set the signals to rms = 1 and also do the smooth crossing between voltages
            logging.debug(f"self._sweep_level_last, self._sweep_signal_rms: {self._sweep_level_last}, {self._sweep_signal_rms}")
            mono_signal_chunk = mono_signal_chunk  / np.exp2(-0.5)\
                * make_fade_window_n(self._sweep_level_last[self._sweep_channel],
                                     self._sweep_signal_rms[self._sweep_channel],
                                     frames,
                                     )
            self._sweep_level_last[self._sweep_channel] = self._sweep_signal_rms[self._sweep_channel]

            # If this was the last fade-out callback and calling back needs to stop
            if self.fade_out_frames["remaining"] <= frames:
                do_callback_stop = True

            # Apply fade-out
            if not np.isnan(self.fade_out_frames["remaining"]):
                fade_start_end_idx = (self.fade_out_frames["remaining"] - self.fade_out_frames["total"],
                                      self.fade_out_frames["remaining"],
                                      )
                fade_out_window = make_fade_window_n(1,
                                                     0,
                                                     frames,
                                                     fade_start_end_idx,
                                                     )
                mono_signal_chunk = mono_signal_chunk * fade_out_window

                logging.debug(f"Remaining/prepared fade out frames: {self.fade_out_frames['remaining']}/{len(fade_out_window)}")
                if self.log_output_signal:
                    self.output_log["fade_out_window"] = np.concatenate([self.output_log["fade_out_window"], fade_out_window])

                self.fade_out_frames["remaining"] -= frames

            else:
                if self.log_output_signal:
                    self.output_log["fade_out_window"] = np.concatenate([self.output_log["fade_out_window"], np.ones(frames) * np.nan])

            # Reset the fade-out counters
            if (self.fade_out_frames["remaining"] <= 0):
                self.reset_fade_out()

            # Make a table with correct rms signal levels
            initial_rms = 1  # rms was made 1 above
            target_rms_levels = np.zeros(self.stream.channels)
            logging.debug(f"_sweep_channel, _sweep_signal_rms: {self._sweep_channel}, {self._sweep_signal_rms}")
            target_rms_levels[self._sweep_channel - 1] = 1
            logging.debug(f"Sweep levels: {target_rms_levels}")

            # Tell Main window which frequency you are at
            if self._sweep_signal_rms[self._sweep_channel] == 0 or target_omega == 0:
                self.sweep_generated.emit(np.nan, self.stream.latency)
            else:
                self.sweep_generated.emit(self._omega_last / 2 / np.pi, self.stream.latency)
            # doesn't work on initiation  # what??

            return mono_signal_chunk, initial_rms, target_rms_levels, do_callback_stop

        except Exception as e:
            logging.critical(
                f"Failed to add {frames} frames during sweep generator callback. Error: {repr(e)}")
            raise sd.CallbackAbort

    def callback(self, indata, frames, time, status):
        """
        Callback function for sounddevice player.
        Initiated wheneversound device runs out of buffer.
        Avoid placing memory allocation or i/o tasks in here.
        """
        logging.debug("")
        logging.debug(f"----Callback for DAC time: {time.outputBufferDacTime}----")
        t1_start = pyt_time.perf_counter_ns()

        if status.output_underflow:
            self.log_through_thread.emit("Buffer underflow. Consider increasing latency settings.")
            # raise sd.CallbackAbort
            # Maybe switch to high latency if this occurs
        elif status and not status.priming_output:
            error_message = f"Unexpected callback status: {status}"
            logging.warning(error_message)

        # Nothing to play
        if (self.play_pos is None) and (np.isnan(self.user_req_alpha) and np.isnan(self.user_req_omega)):
            mono_signal_chunk = np.zeros(frames)          
            logging.debug("Nothing to play for the callback. Put in zeros.")

        elif self.play_pos is not None:
            mono_signal_chunk, initial_rms, target_rms_levels, do_callback_stop = self.callback_for_ugs(frames)

        elif (not np.isnan(self.user_req_alpha)) or (not np.isnan(self.user_req_omega)):
            mono_signal_chunk, initial_rms, target_rms_levels, do_callback_stop = self.callback_for_sweep(frames)

        # Write to sound card
        indata[:frames, :self.stream.channels] = mono_signal_chunk\
            .repeat(self.stream.channels, axis=0)\
            .reshape(frames, self.stream.channels)\
            / initial_rms * np.array(target_rms_levels)  # scale for correct voltage

        # log the output signal
        if self.log_output_signal:
            logging.info(f"Adding to log the signal. Remaining fade frames after this: {self.fade_out_frames['remaining']}")
            self.output_log["time_sig"].extend([float(i) for i in indata[:, 0]])  # only channel 1 is logged

            max_length = int(self.stream.samplerate * 3)
            if len(self.output_log["time_sig"]) > max_length:
                self.output_log["time_sig"] = self.output_log["time_sig"][-max_length:]

            if len(self.output_log["fade_out_window"]) > max_length:
                self.output_log["fade_out_window"] = self.output_log["fade_out_window"][-max_length:]

            if len(self.output_log["fade_in_window"]) > max_length:
                self.output_log["fade_in_window"] = self.output_log["fade_in_window"][-max_length:]

            logging.debug(f"Callback current / buffer DAC time: {time.currentTime} / {time.outputBufferDacTime}")
            logging.debug(f"Calculation / play time: {(pyt_time.perf_counter_ns() - t1_start) / 1e6:.3f} ms / {frames / self.stream.samplerate * 1000:.3f} ms")

        # Playing needs to stop
        if do_callback_stop:
            raise sd.CallbackStop

    @qtc.Slot(dict)
    def generate_sweep(self, **kwargs):
        try:
            target_omega = kwargs.get("target_freq", np.nan) * 2 * np.pi
            alpha = kwargs.get("alpha", np.nan)

            if not (np.nan in (target_omega, alpha)):
                raise KeyError("Cannot define both frequency and angular acceleration.")

            if all([val == np.nan for val in (target_omega, alpha)]):
                raise ValueError("What do I play?? You need to define frequency or angular acceleration.")

            # define acceleration or frequency. so sweep should happen now.
            self.user_req_omega, self.user_req_alpha = target_omega, alpha

            # the voltage is being set separetely with a signal and slot "set_sweep_level"

            # If there is no sweep stream ongoing currently
            if (not hasattr(self, "stream")) or (not self.stream.active) or self.play_pos:
                stream_settings = {"samplerate": settings.sweep_sample_rate,
                                   "channels": settings.channel_count,
                                   "latency": settings.stream_latency,
                                   }
                self._initiate_stream(stream_settings)
                logging.info("Sweep stream started.")
                self.stream.start()

        except Exception as e:
            self.signal_exception.emit(str(e))
            logging.critical(f"Sweep generator failed. {e}")
            self.stream.close(ignore_errors=True)

    @qtc.Slot(dict, dict)
    def ugs_play(self, stream_settings, play_kwargs):
        "UGS means 'user generated signal'"
        try:
            if hasattr(self, "stream"):
                # Already doing a usg play
                self.stop_play()
                while self.stream.active:
                    pass

            self._initiate_stream(stream_settings)

            self.user_gen_signal = play_kwargs["signal_object"]
            self.set_ugs_play_levels(play_kwargs["requested_voltages"])

            self.is_play_in_loop = play_kwargs["loop"]
            self.play_pos = 0

            status_info_text = "---- Playing ----" if not play_kwargs["loop"] else "---- Playing in loop ----"
            now = datetime.now()
            status_info_text += f"\nLocal time at start: {now.strftime('%B %d, %H:%M:%S')}"

            for cn in range(1, stream_settings["channels"] + 1):
                channel_rms = self._ugs_play_voltages[cn]
                if channel_rms > 0:
                    status_info_text += (
                        f"\n\nChannel {cn}:"
                        f"\nAverage output: {channel_rms:.5g} Vrms"
                        f"\nPeak output: {channel_rms * self.user_gen_signal.CF:.5g} V"
                        f"\nSystem gain: {10**(settings.system_gains[cn-1]/20):.5g}x, {settings.system_gains[cn-1]:.4g}dB"
                    )
                else:
                    status_info_text += (
                        f"\n\nChannel {cn}:\nMuted."
                    )

            self.stream.start()

            self.play_started.emit(status_info_text)

            logging.debug(f"Stream started with block sizes {self.stream.blocksize}.")

        except Exception as e:
            self.signal_exception.emit(repr(e))
            logging.error(f"Play_once failed during start. {e}")
            self.stream.close(ignore_errors=True)

    @qtc.Slot()
    def stop_play(self):
        # This is a blocking function due to the while loop below
        if hasattr(self, "stream") and self.stream.active:
            if np.isnan(self.fade_out_frames["remaining"]):
                self.fade_out_frames = {"remaining": self.fade_window_size,
                                        "total": self.fade_window_size,
                                        "stop_after": True,
                                        }
            else:
                pass
            logging.debug("Stop fadeout initiated.")

        else:
            self.play_stopped.emit("Stopped.")
            logging.debug("Stream was not active when stop was requested.")

    @qtc.Slot(float)
    def set_ugs_play_levels(self, voltage_dict: dict) -> None:
        """Creates a dictionary for rms voltages of user generated signal player.
        The keys of dictionary are user friendly channel names, starting from 1.
        """
        user_req_voltages = {}

        for cn in np.arange(1, settings.channel_count + 1):
            if isinstance(voltage_dict, dict) and cn in voltage_dict:
                user_req_voltages[cn] = float(voltage_dict[cn])
            else:
                user_req_voltages[cn] = 0.

        self._ugs_play_voltages = user_req_voltages
        self._ugs_play_signal_rms = self.calculate_digital_signal_rms(user_req_voltages, self.user_gen_signal.CF)
        logging.debug("User generated signal play levels updated in player.")

    @qtc.Slot(float)
    def set_sweep_level(self, voltage: float) -> None:
        """
        Receive requested voltage output and set the correct gain in a dictionary attribute.
        """
        rms_required = self.calculate_digital_signal_rms({self._sweep_channel: voltage}, np.exp2(0.5))[self._sweep_channel]
        self._sweep_voltage = float(voltage)

        if not hasattr(self, "_sweep_signal_rms"):
            self._sweep_signal_rms = {}
        self._sweep_signal_rms[self._sweep_channel] = rms_required

        logging.debug("Sweep level updated in player.")

        # not handling exceptions within this function. risky.

    @qtc.Slot(int)
    def set_sweep_channel(self, channel):
        """Sets channel for sweep as an integer value.
        The integer is user friendly, starting from 1.
        """
        if hasattr(self, "stream") and self.stream.active:
            self.stop_play()  # for user safety
            while self.stream.active:
                pass  # wait until play ends

        self._sweep_channel = int(channel)
        self.set_sweep_level(self._sweep_voltage)

        logging.debug("Sweep channel updated in player.")


class FileWriter(qtc.QThread):
    file_write_successful = qtc.Signal(str)
    file_write_busy = qtc.Signal(str)
    file_write_fail = qtc.Signal(str)

    def __init__(self, parent, generated_signal, **kwargs):  # why is it necessary to pass app to this??
        super().__init__(parent=parent)
        self.generated_signal = generated_signal
        self.kwargs = kwargs

    def run(self):
        self.file_write_busy.emit("Choose file name...")
        logging.debug(f"Writer thread started with params: {self.generated_signal.analysis}, {self.kwargs}")
        channels = self.generated_signal.channel_count()

        try:
            sf_args = ["w"]
            sf_kwargs = {"samplerate": self.generated_signal.FS,
                         "format": self.kwargs["file_format"],
                         "channels": channels,
                         }

            if self.kwargs["file_format"] in ["FLAC", "WAV"]:
                sf_kwargs["subtype"] = "PCM_24"

            # Apply gain to each channel
            time_sig_with_gain = np.empty((self.generated_signal.time_sig.shape[0], channels))
            for i in range(channels):
                time_sig_with_gain[:, i] = self.generated_signal.time_sig * self.kwargs["file_rms"]

            with sf.SoundFile(self.kwargs["file_name"], *sf_args, **sf_kwargs) as sound_file:
                file_info = ("Parameters: "
                             + str(sound_file)
                             + "\n\nFile RMS level: "
                             + f"{self.kwargs['file_rms']:.5g}x, {20*np.log10(self.kwargs['file_rms']):.4g}dB"
                             + "\nFile peak level: "
                             + f"{self.kwargs['file_rms'] * self.generated_signal.CF:.5g}x,  {20*np.log10(self.kwargs['file_rms'] * self.generated_signal.CF):.4g}dB"
                             )
                self.file_write_busy.emit("Writing with parameters:\n\n" + file_info)
                sound_file.write(time_sig_with_gain)
                sound_file.flush()
                self.file_write_successful.emit("Write successful.\n\n" +
                                                file_info + "\n\n\nStopped file writer.")
                logging.info(f"File '{self.kwargs['file_name']}' write successful.")
        except Exception as e:
            self.file_write_fail.emit("Error during file write: " + str(e))
            raise e


class PlayerLogger(qtc.QThread):
    def __init__(self):
        super().__init__()
        # self.setPriority(qtc.QThread.LowestPriority)

    @qtc.Slot()
    def log(self, message):
        logging.info(message)


@dataclass
class Settings:
    system_gains: tuple = tuple([40] * 10)  # starts from 0. all other channel numbers in app start from 1 and use dictionaries
    app_name: str = app_definitions["app_name"]
    author: str = app_definitions["author"]
    author_short: str = app_definitions["author_short"]
    version: str = app_definitions["version"]
    preferred_device: str = "Windows DirectSound - Primary Sound Driver"
    amp_peak: float = 99.
    max_channel_count: int = 10
    channel_count: int = 2
    sweep_sample_rate: int = 44100
    stream_latency: str = "high"
    file_folder: str = ""

    def __post_init__(self):
        settings_storage_title = self.app_name + " - " + (self.version.split(".")[0] if "." in self.version else "")
        self.settings_sys = qtc.QSettings(
            self.author_short, settings_storage_title)
        self.read_all_from_registry()

    def update_attr(self, attr_name, new_val):
        if not new_val:
            return
        elif type(getattr(self, attr_name)) != type(new_val):
            logging.warning(f"Settings.update_attr: Received value type {type(new_val)} does not match the original type {type(getattr(self, attr_name))}"
                            f"\nValue: {new_val}")

        setattr(self, attr_name, new_val)
        self.settings_sys.setValue(attr_name, getattr(self, attr_name))

    def write_all_to_registry(self):
        for field in fields(self):
            value = getattr(self, field.name)
            
            # convert tuples to list for Qt compatibility
            value = list(value) if isinstance(value, tuple) else value

            self.settings_sys.setValue(field.name, value)

    def read_all_from_registry(self):
        for field in fields(self):

            try:
                value_raw = self.settings_sys.value(field.name, field.default)
                value = field.type(value_raw)
            except (TypeError, ValueError):
                value = field.default
    
            setattr(self, field.name, value)

    def as_dict(self):
        settings = {}
        for field in fields(self):
            settings[field] = getattr(self, field.name)
        return settings

    def __repr__(self):
        return str(self.as_dict())


class MainWindow(qtw.QMainWindow):
    global settings, app_definitions

    gen_signal_not_ready = qtc.Signal(str)
    gen_parameters_changed = qtc.Signal()
    play_parameters_changed = qtc.Signal()
    sys_parameters_changed = qtc.Signal()

    def __init__(self, app):  # is this app thing really necessary?
        """MainWindow constructor"""
        super().__init__()

        # Main UI code goes here
        self.setMinimumWidth(1024)
        self.setWindowTitle(app_definitions["app_name"])


        # ---- 'Generate' tab
        signal_type_selector = qtw.QComboBox()

        signal_type_selector.addItems(["Pink noise",
                                       "White noise",
                                       "IEC 268",
                                       "Sine wave",
                                       "Imported"])
        signal_type_selector.activated.connect(self.gen_parameters_changed)  # int

        frequency_widget = qtw.QDoubleSpinBox(Minimum=1,
                                              Maximum=999999,
                                              Value=500,
                                              Decimals=1,
                                              SingleStep=1,
                                              ToolTip="frequency in Hz",
                                              )
        frequency_widget.setEnabled(False)

        frequency_widget.valueChanged.connect(self.gen_parameters_changed)  # float

        compression_widget = qtw.QDoubleSpinBox(Minimum=-10,
                                                Maximum=10,
                                                SingleStep=0.05,
                                                )
        compression_widget.setToolTip("a > 0 is expansion, a = 0 is no change, a < 0 is compression."
                                      "\nTry different values to reach the crest factor that you aim."
                                      )
        compression_widget.valueChanged.connect(self.gen_parameters_changed)

        duration_widget = qtw.QDoubleSpinBox(Minimum=1,
                                             Maximum=60*10,
                                             Value=10,
                                             Decimals=2,
                                             ToolTip="Duration of generated signal in seconds."
                                                     "\nMaximum allowed value is 600."
                                                     "\nWarning: Long signals with high sampling rates "
                                                     "will take a long time to generate!"
                                             )
        duration_widget.valueChanged.connect(self.gen_parameters_changed)

        sample_rate_selector = qtw.QComboBox()
        for i in [22050, 44100, 48000, 96000]:
            sample_rate_selector.addItem(str(i), i)
        sample_rate_selector.setCurrentIndex(2)
        sample_rate_selector.currentTextChanged.connect(self.gen_parameters_changed)

        # Filters
        self.no_of_filters = 8

        class Filter:
            def __init__(self, parent):
                self.widgets = {"type": qtw.QComboBox(),
                                "frequency": qtw.QSpinBox(Minimum=1,
                                                          Maximum=999999,
                                                          Value=1000,
                                                          ),
                                "order": qtw.QComboBox(),
                                }
                self.widgets["frequency"].valueChanged.connect(parent.gen_parameters_changed)
                for i in [1, 2, 4]:
                    self.widgets["order"].addItem(str(i), i)
                self.widgets["order"].setCurrentIndex(1)
                self.widgets["order"].currentTextChanged.connect(parent.gen_parameters_changed)

                # self.widgets["type"].addItems(["Disabled",
                #                                "HP (zero phase)",
                #                                "LP (zero phase)",
                #                                "HP",
                #                                "LP",
                #                                ])
                # disabled zero phase due to bug. issue open in GitHub:
                # https://github.com/python-acoustics/python-acoustics/issues/240

                self.widgets["type"].addItems(["Disabled", "HP", "LP"])
                self.widgets["type"].currentTextChanged.connect(parent.gen_parameters_changed)
                self.layout = qtw.QHBoxLayout()

                for filter in self.widgets.values():
                    self.layout.addWidget(filter)

        filts_layout, filts_widgets = [None] * self.no_of_filters, [None] * self.no_of_filters

        for i in range(self.no_of_filters):
            filter = Filter(parent=self)
            filts_layout[i], filts_widgets[i] = filter.layout, filter.widgets

        # Generator parameters form
        gen_form_layout = qtw.QFormLayout()
        gen_form_layout.addRow("Signal type", signal_type_selector)
        gen_form_layout.addRow("Frequency", frequency_widget)
        for i in range(self.no_of_filters):
            gen_form_layout.addRow(f"Filter {i + 1}", filts_layout[i])
        gen_form_layout.addRow("Compression", compression_widget)
        gen_form_layout.addRow("Duration", duration_widget)
        gen_form_layout.addRow("Sample rate", sample_rate_selector)

        # 'Generate' button
        generate_button = qtw.QPushButton("Generate",
                                          MinimumHeight=40,
                                          )

        # Make the total layout and widget of generator group
        generate_group = qtw.QWidget()
        generate_group_layout = qtw.QVBoxLayout()
        generate_group.setLayout(generate_group_layout)

        # Add the widgets, layouts
        generate_group_layout.addLayout(gen_form_layout)
        generate_group_layout.addSpacing(10)
        generate_group_layout.addWidget(generate_button)

        # ---- 'Play' tab
        sys_gain_adjust_button = qtw.QPushButton("Define system gain parameters")

        # Form for levels
        level_widgets = {}
        max_channel_count = int(settings.max_channel_count)
        for i in range(1, max_channel_count + 1):
            level_widgets[i] = qtw.QDoubleSpinBox(Minimum=0,
                                                  Maximum=999,
                                                  SingleStep=0.1,
                                                  Value=0,
                                                  ToolTip="\n".join(["in Vrms, requested output voltage."])
                                                  )
            level_widgets[i].valueChanged.connect(self.play_parameters_changed)

        speaker_nominal_impedance_widget = qtw.QDoubleSpinBox(Minimum=0.01,
                                                              Maximum=999,
                                                              SingleStep=0.1,
                                                              Value=4,
                                                              ToolTip="in ohms")

        speaker_nominal_power_widget = qtw.QLabel()

        play_in_loop_widget = qtw.QCheckBox(checked=True)
        play_in_loop_widget.stateChanged.connect(self.play_parameters_changed)

        # Player parameters form
        player_params_widget = qtw.QWidget()
        play_params_form_layout = qtw.QFormLayout()
        play_params_form_layout.setContentsMargins(0, 0, 0, 0)
        player_params_widget.setLayout(play_params_form_layout)

        play_params_form_layout.addWidget(sys_gain_adjust_button)
        play_params_form_layout.addWidget(qtw.QFrame(FrameShape=qtw.QFrame.HLine,
                                                     FrameShadow=qtw.QFrame.Sunken),
                                          )

        for i in level_widgets.keys():
            play_params_form_layout.addRow(f"Output voltage for Ch. {i}", level_widgets[i])
        play_params_form_layout.addRow("Play in loop", play_in_loop_widget)
        play_params_form_layout.addRow("Speaker nominal impedance", speaker_nominal_impedance_widget)
        play_params_form_layout.addRow("Nominal power at speaker", speaker_nominal_power_widget)

        # Buttons
        play_button = qtw.QPushButton("Play",
                                      MinimumHeight=40,
                                      )
        stop_button = qtw.QPushButton("Stop",
                                      MinimumHeight=40,
                                      )
        player_buttons_layout = qtw.QHBoxLayout()
        player_buttons_layout.addWidget(play_button)
        player_buttons_layout.addWidget(stop_button)

        # Sound device info
        sound_device_info_widget = qtw.QTextEdit(readOnly=True)

        # Make the total layout and widget of generator group
        player_group = qtw.QWidget()
        play_group_layout = qtw.QVBoxLayout()
        player_group.setLayout(play_group_layout)

        # Add the widgets, layouts
        play_group_layout.addWidget(player_params_widget)
        play_group_layout.addSpacing(10)
        play_group_layout.addWidget(qtw.QLabel("<b>Sound Device</b>"),
                                    alignment=qtc.Qt.AlignHCenter,
                                    )
        play_group_layout.addWidget(sound_device_info_widget)
        play_group_layout.addSpacing(10)
        play_group_layout.addLayout(player_buttons_layout)

        # ---- 'Sweep' widget
        sweep_group = qtw.QWidget()
        sweep_group_layout = qtw.QHBoxLayout()
        sweep_group_layout.setContentsMargins(20, 20, 20, 20)
        sweep_group.setLayout(sweep_group_layout)
        sweep_group.setFont(qtg.QFont("AnyStyle", 12))

        # Freq section
        freq_section = qtw.QVBoxLayout()
        freq_section.setContentsMargins(60, -1, 60, -1)
        freq_display = qtw.QLCDNumber(DigitCount=5,
                                      )
        freq_display.display(np.nan)

        freq_display.setMaximumHeight(180)

        freq_dial_label = qtw.QLabel("Frequency")
        freq_dial_label.setSizePolicy(qtw.QSizePolicy.Preferred, qtw.QSizePolicy.Maximum)

        freq_dial = qtw.QDial(Minimum=0,
                              Maximum=4095,
                              )

        # Freq section layout
        freq_section.addWidget(freq_dial_label, 0, alignment=qtc.Qt.AlignHCenter)
        freq_section.addWidget(freq_display, 2)
        freq_section.addWidget(freq_dial, 10)

        # Other settings section
        other_settings_section = qtw.QVBoxLayout()
        other_settings_section.setContentsMargins(60, 0, 60, 0)

        sweep_status = qtw.QLabel("Waiting",
                                  Font=qtg.QFont("AnyStyle", 14),
                                  alignment=qtc.Qt.AlignCenter,
                                  # MinimumHeight=60,
                                  )

        voltage_spin_box_label = qtw.QLabel("Voltage")
        voltage_spin_box_label.setSizePolicy(qtw.QSizePolicy.Preferred, qtw.QSizePolicy.Maximum)
        voltage_spin_box = qtw.QDoubleSpinBox(Font=qtg.QFont("AnyStyle", 18))

        # voltage_spin_box.lineEdit().setReadOnly(True)  # for safety
        voltage_spin_box.setSingleStep(0.1)

        sweep_channel_label = qtw.QLabel("Channel")
        sweep_channel_label.setSizePolicy(qtw.QSizePolicy.Preferred, qtw.QSizePolicy.Maximum)
        sweep_channel = qtw.QSpinBox(Maximum=int(settings.channel_count),
                                     Font=qtg.QFont("AnyStyle", 18),
                                     Minimum=1,
                                     )
        sweep_stop_button = qtw.QPushButton("Stop",
                                            MinimumSize=qtc.QSize(220, 90),
                                            # Font=qtg.QFont("AnyStyle", 12),
                                            )
        sys_gain_adjust_button_2 = qtw.QPushButton("Define system gain parameters",
                                                   MinimumSize=qtc.QSize(220, 30),
                                                   Font=qtg.QFont("AnyStyle", 8),
                                                   )

        # Other settings section layout
        # Message section
        other_settings_section.addStretch(2)
        other_settings_section.addWidget(sweep_status, alignment=qtc.Qt.AlignHCenter)

        # Separator
        other_settings_section.addStretch(2)
        other_settings_section.addWidget(qtw.QFrame(FrameShape=qtw.QFrame.HLine,
                                                    FrameShadow=qtw.QFrame.Sunken)
                                         )
        other_settings_section.addStretch(2)

        # User adjustments section
        other_settings_section.addWidget(voltage_spin_box_label, 0, alignment=qtc.Qt.AlignHCenter)
        other_settings_section.addWidget(voltage_spin_box, 5, alignment=qtc.Qt.AlignHCenter)
        other_settings_section.addStretch(1)
        other_settings_section.addWidget(sweep_channel_label, 0, alignment=qtc.Qt.AlignHCenter)
        other_settings_section.addWidget(sweep_channel, 5, alignment=qtc.Qt.AlignHCenter)
        other_settings_section.addStretch(1)
        other_settings_section.addWidget(sweep_stop_button, 5, alignment=qtc.Qt.AlignHCenter)

        # Separator
        other_settings_section.addStretch(2)
        other_settings_section.addWidget(qtw.QFrame(FrameShape=qtw.QFrame.HLine,
                                                    FrameShadow=qtw.QFrame.Sunken)
                                         )
        other_settings_section.addStretch(1)

        # Sys gain button
        other_settings_section.addWidget(sys_gain_adjust_button_2, 1, alignment=qtc.Qt.AlignHCenter)
        other_settings_section.addStretch(1)

        # Total layout
        sweep_group_layout.addLayout(freq_section, 4)
        sweep_group_layout.addWidget(qtw.QFrame(FrameShape=qtw.QFrame.VLine,
                                                FrameShadow=qtw.QFrame.Sunken)
                                     )
        sweep_group_layout.addLayout(other_settings_section, 2)

        # ---- 'Write file' tab
        file_rms_title = qtw.QLabel("File RMS level (dBFS)\n20*log10(A) + B",
                                    alignment=qtc.Qt.AlignHCenter)
        self.file_rms_multiplier_widget = qtw.QDoubleSpinBox(Minimum=0.0001,
                                                             Value=1)
        self.file_rms_db_widget = qtw.QDoubleSpinBox(Minimum=-199,
                                                     Value=-20)
        self.file_format_widget = qtw.QComboBox()
        self.file_format_widget.addItems(["FLAC", "WAV"])
        # vorbis ogg removed. it was causing crash with cx_freeze.

        # File write info
        write_file_info_widget = qtw.QTextEdit(readOnly=True,
                                               Text="Inactive")

        # Write file parameters form
        write_file_form_layout = qtw.QFormLayout()
        write_file_form_layout.addWidget(file_rms_title)
        write_file_form_layout.addRow("A (multiplier)", self.file_rms_multiplier_widget)
        write_file_form_layout.addRow("B (dB)", self.file_rms_db_widget)
        write_file_form_layout.addRow("File format", self.file_format_widget)

        # 'Write to file' button
        write_file_button = qtw.QPushButton("Save to file")
        write_file_button.setMinimumHeight(40)

        # Make the total layout and widget of generator group
        write_file_group = qtw.QWidget()
        write_file_group_layout = qtw.QVBoxLayout()
        write_file_group.setLayout(write_file_group_layout)

        write_file_group_layout.addLayout(write_file_form_layout)
        write_file_group_layout.addSpacing(10)
        write_file_group_layout.addWidget(qtw.QLabel("<b>File Writer</b>"),
                                          alignment=qtc.Qt.AlignHCenter)
        write_file_group_layout.addWidget(write_file_info_widget)
        write_file_group_layout.addSpacing(10)
        write_file_group_layout.addWidget(write_file_button)

        # Message box widgets
        generated_signal_info_widget = qtw.QTextEdit(readOnly=True)
        generated_signal_info_widget.setSizePolicy(qtw.QSizePolicy.MinimumExpanding,
                                                   qtw.QSizePolicy.Preferred)
        # Is there a way to set the size policies with constructor arguments?

        status_info_widget = qtw.QTextEdit(readOnly=True)
        status_info_widget.setSizePolicy(qtw.QSizePolicy.MinimumExpanding,
                                         qtw.QSizePolicy.Preferred)

        # ---- About tab
        about_group = qtw.QLabel(alignment=qtc.Qt.AlignCenter)

        about_text = "\n".join([
        f"{app_definitions['description']}",
        f"Version: {app_definitions['version']}",
        "",
        f"Copyright (C) 2023 - {app_definitions['author']}",
        f"{app_definitions['website']}",
        f"{app_definitions['email']}",
        "",
        "This program is free software: you can redistribute it and/or modify",
        "it under the terms of the GNU General Public License as published by",
        "the Free Software Foundation, either version 3 of the License, or",
        "(at your option) any later version.",
        "",
        "This program is distributed in the hope that it will be useful,",
        "but WITHOUT ANY WARRANTY; without even the implied warranty of",
        "MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the",
        "GNU General Public License for more details.",
        "",
        "You should have received a copy of the GNU General Public License",
        "along with this program.  If not, see <https://www.gnu.org/licenses/>.",
        "",
        "This software uses Qt for Python under the GPLv3 license.",
        "https://www.qt.io/",
        "",
        "See 'requirements.txt' for an extensive list of Python libraries used.",
        ])
        about_group.setText(about_text)

        # ---- Layout of main window
        # Layout left side (tabs)
        mw_left_widget = qtw.QTabWidget()
        mw_left_widget.addTab(generate_group, "Generator")
        mw_left_widget.addTab(player_group, "Player")
        mw_left_widget.addTab(write_file_group, "Write file")
        mw_left_widget.addTab(sweep_group, "Sweep generator")
        mw_left_widget.addTab(about_group, "About")
        mw_left_widget.setMinimumWidth(500)
        mw_left_widget.setSizePolicy(qtw.QSizePolicy.Minimum, qtw.QSizePolicy.Minimum)

        # Layout Right side
        mw_right_widget = qtw.QWidget()
        mw_right_layout = qtw.QVBoxLayout()
        mw_right_layout.addWidget(qtw.QLabel("<b>Generated Signal</b>"),
                                  alignment=qtc.Qt.AlignHCenter,
                                  )
        mw_right_layout.addWidget(generated_signal_info_widget)
        mw_right_widget.setLayout(mw_right_layout)

        mpl_widget = MatplotlibWidget(self)
        mpl_widget.setMinimumWidth(400)
        mpl_widget.canvas.setSizePolicy(qtw.QSizePolicy.MinimumExpanding, qtw.QSizePolicy.Expanding)
        mw_right_layout.addWidget(mpl_widget)
        mw_right_layout.addWidget(qtw.QFrame(FrameShape=qtw.QFrame.HLine,
                                             FrameShadow=qtw.QFrame.Sunken),
                                  )

        mw_right_layout.addWidget(qtw.QLabel("<b>Player status</b>"),
                                  alignment=qtc.Qt.AlignHCenter,
                                  )
        mw_right_layout.addWidget(status_info_widget)

        # Layout Top Level
        mw_center_widget = qtw.QWidget()
        mw_center_layout = qtw.QHBoxLayout()
        mw_center_widget.setLayout(mw_center_layout)
        self.setCentralWidget(mw_center_widget)

        mw_center_layout.addWidget(mw_left_widget)
        mw_center_separator = qtw.QFrame(FrameShape=qtw.QFrame.VLine,
                                         FrameShadow=qtw.QFrame.Sunken
                                         )
        mw_center_layout.addWidget(mw_center_separator)
        mw_center_layout.addWidget(mw_right_widget)

        # ---- Start generator and player threads
        self.generator = Generator()
        self.generator_thread = qtc.QThread()
        self.generator.moveToThread(self.generator_thread)
        self.generator_thread.start(qtc.QThread.LowPriority)

        self.player_logger = PlayerLogger()
        self.player_logger.start(qtc.QThread.LowestPriority)

        self.player_thread = qtc.QThread()
        self.player = Player()
        self.player.moveToThread(self.player_thread)
        self.player_thread.start(qtc.QThread.HighPriority)

        qtw.QApplication.instance().aboutToQuit.connect(self.player.stop_play)
        qtw.QApplication.instance().aboutToQuit.connect(self.player_thread.quit)
        qtw.QApplication.instance().aboutToQuit.connect(self.generator_thread.quit)
        qtw.QApplication.instance().aboutToQuit.connect(self.player_logger.quit)

        # ---- Functions triggered by user through the GUI
        def gain_and_levels_button_clicked():
            sys_gain_widget = SysGainAndLevelsPopup()
            sys_gain_widget.user_changed_sys_params_signal.connect(self.sys_parameters_changed)
            sys_gain_widget.exec()

        def play_clicked():
            if hasattr(self, "generated_signal") and isinstance(self.generated_signal, TestSignal):
                # Params to initiate string
                channel_count = settings.channel_count
                channels_range = range(1, channel_count + 1)

                stream_settings = {"samplerate": sample_rate_selector.currentData(),
                                   "channels": channel_count,
                                   "latency": settings.stream_latency,
                                   }

                # Params to play signal
                requested_voltages = {n_c: level_widgets[n_c].value() for n_c in channels_range}

                play_kwargs = {
                    "signal_object": self.generated_signal,
                    "loop": play_in_loop_widget.checkState(),
                    "requested_voltages": requested_voltages,
                    }

                self.player.ugs_play(stream_settings, play_kwargs)

            else:
                error_text = "No signal found to play."
                informative_text = "Generate a signal using the generator tab."
                PopupError(error_text, informative_text)

        def generate_clicked():
            try:
                # Make the signal
                sig_type = signal_type_selector.currentText()
                kwargs = {"filters": filts_widgets,
                          "frequency": frequency_widget.value(),
                          "compression": compression_widget.value(),
                          "T": duration_widget.value(),
                          "FS": sample_rate_selector.currentData(),
                          }
                if sig_type == "Imported":
                    self.generator.process_imported_file("Reuse existing", kwargs)
                else:
                    self.generator.generate_ugs(sig_type, kwargs)
            except Exception as e:
                error_text = "Unable to place generator request in the generator thread."
                logging.critical(str(e))
                PopupError(error_text, str(e))

        def write_file_clicked():
            if not self.generated_signal:
                error_text = "No signal found to write."
                informative_text = "Generate a signal using the generator tab."
                PopupError(error_text, informative_text)
                return
            write_args = {"file_format": self.file_format_widget.currentText(),
                          "file_rms": 10**(self.file_rms_db_widget.value() / 20) * self.file_rms_multiplier_widget.value(),
                          }
            if write_args["file_rms"] * self.generated_signal.CF > 1:
                error_text = "Current settings will cause digital clipping."
                informative_text = "Reduce target RMS voltage and/or crest factor.\nMake sure system gain is entered correctly and increase amplifier gain if necessary."
                PopupError(error_text, informative_text)
                return
            try:
                self.player.stop_play()
                # pyt_time.sleep(0.001)
                file_filters = {"FLAC": "FLAC files (*.flac)",
                                "WAV": "Wave files (*.wav)",
                                "OGG": "Vorbis files (*.ogg)",
                                }
                write_args["file_name"] = qtw.QFileDialog.getSaveFileName(None, "Save audio signal in file...",
                                                                          os.getcwd(),
                                                                          file_filters[write_args["file_format"]],
                                                                          "",
                                                                          )[0]
                # is this app thing really necessary? why not use qtw.QApplication.instance()
                writer = FileWriter(app, self.generated_signal, **write_args)
                writer.file_write_successful.connect(write_file_info_widget.setText)
                writer.file_write_busy.connect(write_file_info_widget.setText)
                writer.file_write_fail.connect(write_file_info_widget.setText)
                writer.finished.connect(lambda: logging.debug("Finished thread file writer"))
                writer.start()
            except Exception as e:
                error_text = "File writer failed."
                PopupError(error_text, str(e))

        def choose_import_file():
            try:
                self.player.stop_play()

                # Wait if playback is going on. This call does file access and can cause
                # buffer underrun in player callback
                for timer in range(10):
                    if self.player.is_active:
                        qtc.QThread.msleep(100)
                    if timer == 99:
                        raise RuntimeError("Could not stop player thread.")
                    else:
                        qtc.QThread.msleep(100)
                        break

                # add functionality for remembering latest file_folder
                file_folder = settings.file_folder
                if not isinstance(file_folder, str) or file_folder == "" or not os.path.exists(file_folder):
                    file_folder = os.getcwd()

                # ask user to pick file
                file_formats = " ".join(["*." + str(suffix).lower() for suffix in sf.available_formats()])
                file_path = qtw.QFileDialog.getOpenFileName(None,
                                                            "Choose audio file to import...",
                                                            file_folder,
                                                            f"Audio files ({file_formats})",
                                                            )[0]
                if file_path:
                    settings.update_attr("file_folder", os.path.dirname(file_path))
                    self.generator.import_file(file_path)
                else:
                    self.gen_signal_not_ready.emit("No file chosen.")
                    self.generator.clear_imported_file()

            except Exception as e:
                error_text = "File import failed."
                PopupError(error_text, str(e))
                settings.update_attr("file_folder", "")

        def generate_sweep(dial_value):

            f_start = 10
            f_end = 2e4
            dial_max_value = 4095
            # freq(dial_value) = 10**(k * dial_value - m)
            try:
                if dial_value == 0:
                    freq_on_dial = 0
                else:
                    k = np.log10(f_end / f_start) / (dial_max_value - 1)
                    m = np.log10(1 / f_start)
                    freq_on_dial = 10**(k * (dial_value - 1) - m)
                self.player.generate_sweep(target_freq=freq_on_dial)

            except Exception as e:
                error_text = "Unable to place sweep generate request in the player thread."
                logging.critical(repr(e))
                PopupError(error_text, repr(e))

        # User changed generator signal type
        def signal_type_selection_changed():
            duration_widget.setEnabled(signal_type_selector.currentText() != "Imported")

            if signal_type_selector.currentText() == "Imported":
                choose_import_file()
        signal_type_selector.activated.connect(signal_type_selection_changed)

        # Disabling voltage widgets for disabled channels
        def disable_voltage_output_widgets_for_inactive_channels():
            for i, level_widget in level_widgets.items():
                level_widget.setEnabled(i <= int(settings.channel_count))
        self.play_parameters_changed.connect(disable_voltage_output_widgets_for_inactive_channels)
        disable_voltage_output_widgets_for_inactive_channels()

        # Disable frequency widget when sine is not selected
        signal_type_selector.currentIndexChanged.connect(
            lambda: frequency_widget.setEnabled(
                signal_type_selector.currentIndex() == 3))

        # Give a crest factor warning when IEC signal is selected
        signal_type_selector.currentIndexChanged.connect(
            lambda: PopupError("IEC 268", "Apply a compression of\nabout -3 to get a crest\nfactor of 2.", title="Warning") if signal_type_selector.currentIndex() == 2 else None)

        # Show speaker nominal powers
        def show_nominal_speaker_power():
            values = [widget.value()**2 / speaker_nominal_impedance_widget.value()
                      for widget in level_widgets.values() if widget.isEnabled() is True]
            speaker_nominal_power_widget.setText(" / ".join([f"{value:.3g} W" for value in values]))

        for widget in level_widgets.values():
            widget.valueChanged.connect(show_nominal_speaker_power)
        show_nominal_speaker_power()

        speaker_nominal_impedance_widget.valueChanged.connect(show_nominal_speaker_power)
        self.play_parameters_changed.connect(show_nominal_speaker_power)

        # Change layout based on chosen tab
        def update_layout_based_on_chosen_tab(current_index):
            if current_index in (3, 4):
                mw_center_separator.hide()
                mw_right_widget.hide()
            else:
                mw_center_separator.show()
                mw_right_widget.show()
        mw_left_widget.currentChanged.connect(update_layout_based_on_chosen_tab)

        # Functionality for frequency sweep tab
        freq_dial.valueChanged.connect(generate_sweep)

        sweep_channel.valueChanged.connect(self.player.set_sweep_channel, qtc.Qt.QueuedConnection)
        sweep_channel.valueChanged.emit(sweep_channel.value())

        voltage_spin_box.valueChanged.connect(self.player.set_sweep_level, qtc.Qt.QueuedConnection)
        voltage_spin_box.valueChanged.emit(voltage_spin_box.value())

        self.player.sweep_generated.connect(freq_display.display, qtc.Qt.QueuedConnection)

        def unavailable_feature():
            error_text = "Feature not implemented yet."
            PopupError(error_text)

        # ---- Connection of pushbuttons
        play_button.clicked.connect(play_clicked)
        stop_button.clicked.connect(self.player.stop_play)
        sweep_stop_button.clicked.connect(self.player.stop_play)
        generate_button.clicked.connect(generate_clicked)
        sys_gain_adjust_button.clicked.connect(gain_and_levels_button_clicked)
        sys_gain_adjust_button_2.clicked.connect(gain_and_levels_button_clicked)
        write_file_button.clicked.connect(write_file_clicked)

        # ---- Functions triggered by threads and logic, not the user

        # A file is imported into the generator successfully
        def generator_thread_file_import_success(imported_signal):
            index_to_set = sample_rate_selector.findData(imported_signal.FS)
            if index_to_set == -1:
                sample_rate_selector.addItem(str(imported_signal.FS), imported_signal.FS)
            else:
                sample_rate_selector.setCurrentIndex(index_to_set)
            duration_widget.setValue(imported_signal.T)
            self.gen_signal_not_ready.emit(f"Imported successfully.\n{imported_signal.raw_import_analysis}"
                                                + "\n\nContinue setting up processing and press 'Generate' when ready.")
        self.generator.file_import_success.connect(generator_thread_file_import_success)

        def update_sweep_info_screen(freq, latency):
            if np.isnan(freq) and not np.isnan(latency):
                sweep_status.setText("Muted")
            elif np.isnan(latency):
                sweep_status.setText("Stopped")
            elif all(isinstance(item, float) for item in [freq, latency]):
                info_text = "Output active"
                # info_text += f"\nLatency: {int(latency * 1000)}ms"
                sweep_status.setText(info_text)
            else:
                sweep_status.setText("Unknown state")
        self.player.sweep_generated.connect(update_sweep_info_screen, qtc.Qt.QueuedConnection)
        self.player.sweep_generator_stopped.connect(lambda: update_sweep_info_screen(0, 0))

        # Log something through the thread
        @qtc.Slot(str)
        def log_with_thread(message):
            self.player_logger.log(f"Player: {message}")
        self.player.log_through_thread.connect(log_with_thread)

        # Logging functionality
        def show_log(log_dict):
            log_win = LogView(log_dict)
            log_win.exec()
        self.player.publish_log.connect(show_log)

        # Output voltage request not feasible
        def impossible_voltage_request_happened_at_sweeper(str):
            voltage_spin_box.setValue(0)  # not very user friendly
        self.player.impossible_voltage_request.connect(impossible_voltage_request_happened_at_sweeper)

        # Update detected sound devices every N seconds
        self.poll_sound_devices_timer = qtc.QTimer()
        self.poll_sound_devices_timer.setInterval(2000)
        self.poll_sound_devices_timer.start()  # priority tanımlayınca interval devre dışı kaldı. garip.

        self.poll_sound_devices_timer.timeout.connect(self.player.poll_sound_devices)
        self.player.signal_sound_devices_polled.connect(sound_device_info_widget.setText)

        # ---- Slots of main window GUI
        @qtc.Slot(TestSignal)
        def gen_signal_ready(generated_signal):
            logging.debug("Main window received signal 'Generated signal ready'")
            try:
                self.generated_signal = generated_signal

                # Update user with the changes
                generator_info_text = self.generated_signal.analysis
                mpl_widget.update_plot(generated_signal)

            except Exception as e:
                self.gen_signal_not_ready.emit(
                    "Failed to receive generated signal from generator thread.\n" + str(e))
            else:  # do this always
                generated_signal_info_widget.setText(generator_info_text)
                generate_group.setEnabled(True)

        self.generator.signal_ready.connect(gen_signal_ready)

        @qtc.Slot(str)
        def gen_signal_not_ready(generator_info_text):
            "Signal not ready"
            generate_group.setEnabled(True)
            mpl_widget.clear_plot()
            self.player.stop_play()
            generated_signal_info_widget.setText(generator_info_text)
            generated_signal_info_widget.repaint()  # why?
            self.generated_signal = None

        self.gen_signal_not_ready.connect(gen_signal_not_ready)

        @qtc.Slot(Exception)
        def generator_exception(e):
            error_text = "Error in signal generator."
            informative_text = str(e)
            gen_signal_not_ready(error_text)
            PopupError(error_text, informative_text)

        self.generator.exception.connect(generator_exception)

        @qtc.Slot(Exception)
        def player_exception(e):
            error_text = "Error in player."
            informative_text = str(e)
            PopupError(error_text, informative_text)

        self.player.signal_exception.connect(player_exception)

        @qtc.Slot(str)
        def gen_signal_busy(generator_info_text):
            "Busy"
            generate_group.setEnabled(False)
            mpl_widget.clear_plot()
            self.player.stop_play()
            generated_signal_info_widget.setText(generator_info_text)
            generated_signal_info_widget.repaint()

        self.generator.busy.connect(gen_signal_busy)

        @qtc.Slot(str)
        def play_stopped(message):
            "User generated signal play stopped"
            player_params_widget.setEnabled(True)
            status_info_widget.setText(message)
        self.player.play_stopped.connect(play_stopped)
        self.player.signal_exception.connect(lambda: play_stopped("Stopped due to error in player."))

        @qtc.Slot(str)
        def play_started(play_info_text):
            "User generated signal play started"
            player_params_widget.setEnabled(False)
            status_info_widget.setText(play_info_text)
            # save_settings()
        self.player.play_started.connect(play_started)

        @qtc.Slot(str)
        def gen_parameters_changed(new_param):
            "When generator parameters changed"
            if not (signal_type_selector.currentText() == "Imported" and not self.generated_signal):
                generator_info_text = f'Parameter changed: {new_param}' + \
                    '\nPress "Generate" to generate signal.'
                self.gen_signal_not_ready.emit(generator_info_text)
        self.gen_parameters_changed.connect(gen_parameters_changed)

        @qtc.Slot()
        def play_parameters_changed_actions():
            "Player tab parameters changed"
            self.player.stop_play()
        self.play_parameters_changed.connect(play_parameters_changed_actions)

        @qtc.Slot()
        def sys_parameters_changed_actions():
            "System parameters changed"
            self.player.stop_play()
            logging.warning("System parameters changed by user.")
            sweep_channel.setMaximum(int(settings.channel_count))
            disable_voltage_output_widgets_for_inactive_channels()
            # setting the maximum value for the sweep voltage spin box here would be nice
            # but it depends on channel so not so simple to do
        self.sys_parameters_changed.connect(sys_parameters_changed_actions)


class MatplotlibWidget(qtw.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        fig = plt.Figure()
        self.canvas = FigureCanvas(fig)
        self.toolbar = NavigationToolbar(self.canvas, self)

        lay = qtw.QVBoxLayout(self)
        lay.addWidget(self.toolbar)
        lay.addWidget(self.canvas)
        lay.setContentsMargins(0, 0, 0, 0)
        self.ax = fig.add_subplot(111)
        fig.tight_layout()

    def calculate_3rd_octave_bands(self, time_sig: np.array, FS) -> tuple:
        sig = time_sig.astype("float32")
        threeoct_freqs = ac.standards.iec_61260_1_2014.NOMINAL_THIRD_OCTAVE_CENTER_FREQUENCIES

        if len(time_sig) <= 2**21:
            return (threeoct_freqs,
                    ac.signal.third_octaves(sig, FS, frequencies=threeoct_freqs)[1] + 20 * np.log10(20e-6),
                    )
        else:
            n_arrays = len(time_sig) // 2**20
            logging.debug(f"Calculating octave bands by dividing signal into {n_arrays} pieces.")
            arrays = np.array_split(time_sig, n_arrays)
            third_oct_pows = np.empty((n_arrays, len(threeoct_freqs)))

            for i, array in enumerate(arrays):
                third_oct_pows[i, :] = 10**(ac.signal.third_octaves(array, FS,
                                                                    frequencies=threeoct_freqs)[1] / 10)
            third_oct_pow_averages = (10 * np.log10(np.average(third_oct_pows, axis=0))) - 94
            return (threeoct_freqs, third_oct_pow_averages)

    @qtc.Slot(TestSignal)
    def update_plot(self, generated_signal):
        self.ax.cla()
        if generated_signal:
            # Power spectrum of signal
            FS = generated_signal.FS
            PowerSpect = signal.welch(generated_signal.time_sig.astype("float32"),
                                      fs=FS,
                                      nperseg=FS/4,  # defines also window size
                                      window="hann",
                                      scaling="spectrum")

            # Power per octave band of signal
            threeoct_freqs, three_oct_power = self.calculate_3rd_octave_bands(generated_signal.time_sig, FS)

            self.ax.semilogx(PowerSpect[0], 10*np.log10(PowerSpect[1]), label="Power spectral density")
            self.ax.step(threeoct_freqs, three_oct_power, where="mid", label="1/3 octave bands")

            self.ax.set_xlim(10, FS/2)
            self.ax.set_ylim(-70, 5)
            self.ax.grid(which='minor', axis='x')
            self.ax.grid(which='major', axis='y')
            self.ax.legend()

        self.canvas.draw()

    def clear_plot(self):
        self.update_plot(None)


def main():
    global settings, app_definition
    settings = Settings(app_definitions["app_name"])

    qapp = qtw.QApplication.instance()
    if not qapp:
        qapp = qtw.QApplication(sys.argv)
        qapp.setWindowIcon(qtg.QIcon(app_definitions["icon_path"]))
    mw = MainWindow(qapp)
    mw.show()
    qapp.exec()


if __name__ == "__main__":
    main()
