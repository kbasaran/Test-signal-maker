# Test Signal Maker - Loudspeaker testing tool
# Copyright (C) 2026 - Kerem Basaran
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

import sys
import os
import time

# https://doc.qt.io/qtforpython/
from PySide6 import QtWidgets as qtw
from PySide6 import QtCore as qtc
from PySide6 import QtGui as qtg

# Set environment variable before importing sounddevice. Value is not important.
os.environ["SD_ENABLE_ASIO"] = "1"
# https://python-sounddevice.readthedocs.io/en/latest/installation.html#asio-support
import sounddevice as sd  # https://python-sounddevice.readthedocs.io

import numpy as np
import soundfile as sf  # https://python-soundfile.readthedocs.io/
from scipy import signal
import copy
from datetime import datetime

# from datetime import date
# today = date.today()
from pathlib import Path

import matplotlib.pyplot as plt  # http://matplotlib.org/
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

from generictools.signal_tools import TestSignal, make_fade_window_n, calculate_3rd_octave_bands
import generictools.personalized_widgets as pwi
from dataclasses import dataclass, fields
import logging
# import multiprocessing

app_definitions = {"app_name": "Test Signal Maker",
                   "version": "0.3.3",
                   "description": "Test Signal Maker - Loudspeaker test signal tool",
                   "copyright": "Copyright (C) 2026 Kerem Basaran",
                   "icon_path": "logo/icon.ico",  # relative posix path
                   "author": "Kerem Basaran",
                   "author_short": "kbasaran",
                   "email": "kbasaran@gmail.com",
                   "website": "https://github.com/kbasaran",
                   }

# uncomment for release candidate builds
# app_definitions["version"] += "rc" + time.strftime("%y%m%d", time.localtime())

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
    def __init__(self, text, informative_text=None, post_action=None, parent=None, title="Error"):
        msg = qtw.QMessageBox(parent=parent)
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
            device_name = device['name']
            data_name = hostapi_name + " - " + device_name
            user_friendly_name = f"{data_name} - {device['max_output_channels']} channels"
            if device["index"] == sd.default.device[1]:
                default_device_user_friendly_name = user_friendly_name
                preferred_device_widget.addItem(user_friendly_name, data_name)  # always add the default device
            elif device["max_output_channels"] > 0 and "WDM" not in hostapi_name and "MME" not in hostapi_name:
                preferred_device_widget.addItem(user_friendly_name, data_name)  # add all others

        preferred_device_index = preferred_device_widget.findData(preferred_device_name)  # -1 needs not found, and empty QComboBox selection
        if preferred_device_index == -1:
            preferred_device_widget.setCurrentText(default_device_user_friendly_name)  # set to the default device
        else:
            preferred_device_widget.setCurrentIndex(preferred_device_index)

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
        play_sample_rate = qtw.QComboBox()
        for val in [44100, 48000, 96000]:
            play_sample_rate.addItem(str(val), val)

        # check which value is stored in settings
        current_val = settings.play_sample_rate
        current_val_idx = play_sample_rate.findData(current_val)

        if current_val_idx == -1:  # if the sapmle rate is not available in current list
            play_sample_rate.setCurrentIndex(1)
        else:
            play_sample_rate.setCurrentIndex(current_val_idx)

        # Stream latency
        stream_latency = qtw.QComboBox()
        stream_latency.addItem("Sound device default: High", "high")
        stream_latency.addItem("Sound device default: Low", "low")
        stream_latency.addItem("User value: Safe - 25ms", "0.025")
        stream_latency.addItem("User value: Very safe - 50ms", "0.05")

        current_val = settings.stream_latency  # always as str
        current_val_idx = stream_latency.findData(current_val)

        if current_val_idx == -1:
            stream_latency.setCurrentIndex(0)
        else:
            stream_latency.setCurrentIndex(current_val_idx)

        # Rest of the layouts
        sys_gain_form_layout.addWidget(qtw.QFrame(FrameShape=qtw.QFrame.HLine, FrameShadow=qtw.QFrame.Sunken))
        sys_gain_form_layout.addRow("Amplifier peak capability (V)", amp_peak_capability_widget)

        sys_gain_form_layout.addWidget(qtw.QFrame(FrameShape=qtw.QFrame.HLine, FrameShadow=qtw.QFrame.Sunken))
        sys_gain_form_layout.addRow("Play sample rate", play_sample_rate)
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
            settings.update("preferred_device", preferred_device_widget.currentData())
            system_gains = list(settings.system_gains)
            for cn in channel_gain_widgets.keys():
                system_gains[cn-1] = channel_gain_widgets[cn].value()
            settings.update("system_gains", tuple(system_gains))
            settings.update("amp_peak", amp_peak_capability_widget.value())
            settings.update("channel_count", number_of_channels_widget.value())
            settings.update("max_channel_count", settings.max_channel_count)  # so this stays fixed, no user option to change it yet
            settings.update("play_sample_rate", play_sample_rate.currentData())
            settings.update("stream_latency", stream_latency.currentData())

            self.user_changed_sys_params_signal.emit()
            self.done(0)
            logger.info("System parameters changed by user.")

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
    signal_ready = qtc.Signal(TestSignal, tuple, tuple)
    file_import_success = qtc.Signal(TestSignal)
    signal_not_ready = qtc.Signal(str)
    exception = qtc.Signal(Exception)
    
    def __init__(self):
        super().__init__()
        self.signal_not_ready.emit("No signal has been generated.")

    @qtc.Slot(str)
    def import_file(self, import_file_path):
        logger.debug(f'Importing file...{import_file_path}')
        self.signal_not_ready.emit("Importing file...")
        try:
            self.imported_signal = TestSignal("Imported",
                                              import_file_path=import_file_path,
                                              import_channel="downmix_all",
                                              )
            self.file_import_success.emit(self.imported_signal)
            logger.debug("Signal imported and params published.")

        except Exception as e:
            # Pop-up
            logger.error("Import failed" + str(e))
            self.exception.emit(e)

    @qtc.Slot()
    def clear_imported_file(self):
        self.imported_signal = None

    @qtc.Slot(str, dict)
    def process_imported_file(self, sig_type, kwargs):
        self.signal_not_ready.emit("Generating from import...")
        try:
            if not hasattr(self, "imported_signal") or not self.imported_signal:
                raise KeyError("No file imported to process.")
            generated_signal = copy.deepcopy(self.imported_signal)
            generated_signal.reuse_existing(**kwargs)
            
            self.signal_not_ready.emit(f"Analyzing signal...")
            power_spectrum, octave_bands = generated_signal.spectrum_analysis()
            
            self.signal_ready.emit(generated_signal, power_spectrum, octave_bands)
            logger.debug("Imported signal has been processed and published.")

        except Exception as e:
            # Pop-up
            logger.error(str(e))
            self.exception.emit(e)

    @qtc.Slot(str, dict)
    def generate_ugs(self, sig_type, kwargs):
        logger.debug(f'Generate ugs "{sig_type}" initiated')
        self.signal_not_ready.emit(f"Generating {sig_type.lower()}...")
        try:
            generated_signal = TestSignal(sig_type, **kwargs)
    
            self.signal_not_ready.emit(f"Analyzing signal...")
            power_spectrum, octave_bands = generated_signal.spectrum_analysis()
            
            self.signal_ready.emit(generated_signal, power_spectrum, octave_bands)
            logger.info(f"Signal with type {sig_type} generated.")

        except Exception as e:
            # Pop-up
            logger.error(str(e))
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
        self.log_output_signal = False  # logs channel 1 and plot it for debugging

        # Inititate attributes
        self._sweep_voltage = 0
        self._sweep_channel = 0
        
        err_message = self.initiate_stream()
        if err_message:
            PopupError(err_message)


    def initiate_stream(self, force_sample_rate=None) -> str:
        "Prepare the stream object with provided settings. Does not start it."
        "Returns error message if initiation fails."

        # Close any existing stream
        if hasattr(self, "stream") and self.stream is not None:
            self.stop_play_blocking()
            self.stream.close()
        
        # target settings
        self._bring_sweep_states_to_zero(settings.channel_count)
        sample_rate = force_sample_rate if force_sample_rate else settings.play_sample_rate
        self.ugs_play_stopwatch = -1.
        try: 
            stream_latency = float(settings.stream_latency)  # for values in s
        except ValueError:
            stream_latency = str(settings.stream_latency)  # for 'high' and 'low'
        
        # define the sound device based on settings and availability
        play_device_idx = self.get_right_sound_device()
        
        # check if settings are valid
        output_settings = {
                            "device": play_device_idx,
                            "samplerate": sample_rate,
                            "channels": settings.channel_count,
                        }

        try:
            # sd.check_output_settings(**output_settings)
            self.stream = sd.OutputStream(callback=self.callback,
                                          finished_callback= self.announce_callback_is_finished,
                                          latency= stream_latency,
                                          **output_settings,
                                          )
            
            # a parameter
            self.fade_window_size = int(self.stream.samplerate // 20)

        except Exception as e:
            self.stream = None
            # PopupError(
            #             "Unable to intiate audio stream.",
            #             informative_text="Please check your device and the number of channels. This is the most likely reason for this error.",
            #             post_action=None,
            #             title="Error",
            #             )
            self.signal_exception.emit("Unable to initilize audio stream. Please check your device and the number of channels settings.")


    def reset_fade_out(self):
        self.fade_out_frames = {"remaining": np.nan,
                                "total": np.nan,
                                "stop_after": None,
                                }

    def announce_callback_is_finished(self):
        if self.play_pos:
            if self.stop_after_seconds:
                self.play_stopped.emit(f"Stopped with timer after {self.stop_after_seconds/60:.1f} minutes.")
            else:
                self.play_stopped.emit("Stopped.")

            self.play_pos = None
            self.reset_fade_out()

        self.user_req_omega, self.user_req_alpha = np.nan, np.nan
        self.sweep_generator_stopped.emit("Stopped")
        self._bring_sweep_states_to_zero(self.stream.channels)
        self.sweep_generated.emit(np.nan, np.nan)
        
        # Debugging
        if self.log_output_signal:
            # this crashes when a stream is present but no signal has been played yet, e.g. in startup
            self.publish_log.emit(self.output_log)
            self.output_log = {"time_sig": [],
                                "fade_out_window": [],
                                "fade_in_window": [],
                                }
        
        logger.info("Callback stopped.")

    def get_right_sound_device(self) -> int:
        "Returns the preferred play sound device index."
        preferred_device_name = settings.preferred_device
        device_name_to_index = {}
        for device in sd.query_devices():
            hostapi_name = sd.query_hostapis(index=device['hostapi'])['name']
            device_name = device['name']
            data_name = hostapi_name + " - " + device_name
            device_name_to_index[data_name] = device["index"]

        play_device_idx = device_name_to_index.get(preferred_device_name, sd.default.device[1])
        return play_device_idx
        # 0 is the recording device, 1 is playback
        # sd.default.device returns (int, int)

    @qtc.Slot()
    def poll_sound_devices(self):
        try:
            # this is invoked regularly to update the current sound device in use
            # without this, when default device is changed in operating system, there is no detection
            # https://github.com/spatialaudio/python-sounddevice/issues/337
            # if not self.stream.active:  # if stream is not active
                # sd._terminate()
                # sd._initialize()
            play_device_idx = self.get_right_sound_device()
            play_device_info = sd.query_devices(play_device_idx)
            # this doesn't update when default sound device is changed in operating system :(
            # thus the trick above
            play_device_summary = f"""Device name: {play_device_info['name']}
--Host API: {sd.query_hostapis()[play_device_info['hostapi']]["name"]}
--Max. output channels: {play_device_info['max_output_channels']}
--Default samplerate: {int(play_device_info['default_samplerate'])}
--Default data type: {sd.default.dtype[1]}
"""
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
            informative_text = ("Increase amplifier gain or reduce target RMS voltage and/or signal crest factor."
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

    def _bring_sweep_states_to_zero(self, channel_count):
        self._omega_last = 0.
        self._theta_last = 0.
        self._sweep_level_last = 0.

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
        try:
            T = t_array[-1]
            n = (omega_end / omega_start)**(1 / T)
            k = omega_start / np.log(n)
            theta_array = (theta_start + k * (np.exp(t_array * np.log(n)) - 1)) % (2 * np.pi)
        except RuntimeWarning:
            return np.zeros(len(t_array)), theta_start, omega_start

        mono_signal_chunk = np.sin(theta_array)
        theta_last = theta_array[-1]
        omega_last = omega_end
        return mono_signal_chunk, theta_last, omega_last

    def callback_for_ugs(self, frames):
        "We are doing a callback for streaming an already generated signal"
        stream_needs_to_stop_now = False
        try:
            # Try to fill the soundcard buffer within this loop
            empty_frames = int(frames)
            mono_signal_chunk = np.empty(frames)

            while empty_frames > 0:
                logger.debug("---Fill cycle---")
                logger.debug(f"Play pos: {self.play_pos}")
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

                logger.debug(f"Fade out frames: {self.fade_out_frames}")

                # reached end of fade-out and not gonna loop, so stop calling back
                if (self.fade_out_frames["remaining"] <= empty_frames) and (not self.is_play_in_loop or self.fade_out_frames["stop_after"]):
                    stream_needs_to_stop_now = True

                # Apply fade-out
                if self.fade_out_frames["remaining"] > 0:
                    fade_start_end_idxs = (self.fade_out_frames["remaining"] - self.fade_out_frames["total"],
                                          self.fade_out_frames["remaining"],
                                          )
                    fade_out_window = make_fade_window_n(1,
                                                         0,
                                                         number_of_samples_to_write,
                                                         fade_start_end_idxs,
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
                    fade_start_end_idxs = (-self.play_pos,
                                          self.fade_window_size - self.play_pos,
                                          )
                    fade_in_window = make_fade_window_n(0,
                                                        1,
                                                        number_of_samples_to_write,
                                                        fade_start_end_idxs,
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
            ugs_play_rms_levels = np.empty(self.stream.channels)
            for channel in range(1, self.stream.channels + 1):
                ugs_play_rms_levels[channel - 1] = self._ugs_play_signal_rms[channel]
            if self.log_output_signal:
                logger.debug(f"User generated signal play levels: {ugs_play_rms_levels:.4f}")

            if self.ugs_play_stopwatch == -1.:
                self.log_through_thread.emit(f"Started with: {self._ugs_play_voltages}Vrms")
            self.ugs_play_stopwatch = time.time()
            if time.time() > self.ugs_play_stopwatch + 60 * 60:  # every hour
                self.log_through_thread.emit(f"Ongoing with: {self._ugs_play_voltages}Vrms")
                self.ugs_play_stopwatch = time.time()
            return mono_signal_chunk, initial_rms, ugs_play_rms_levels, stream_needs_to_stop_now

        except Exception as e:
            logger.critical(
                f"Failed to add {frames} frames during usg callback." +
                f"\nPosition: {self.play_pos}/{len(self.user_gen_signal.time_sig)}. Error: {str(e)}")
            raise sd.CallbackAbort  # why?

    def callback_for_sweep(self, frames):
        "We are doing a frequency generator callback"
        stream_needs_to_stop_now = False

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
                logger.debug("Callback case exponential")
                mono_signal_chunk, self._theta_last, self._omega_last =\
                    self.calculate_exp_sweep(t_array,
                                             self._theta_last,
                                             self._omega_last,
                                             target_omega,
                                             )
                # output should be faded out to zero if target is 0Hz.

            # Need to be quiet
            elif target_omega == 0 and self._omega_last == 0:
                logger.debug("Callback case zero output")
                # Otherwise it clicks.
                mono_signal_chunk = np.zeros(frames)

            # Linear sweep is necessary
            else:
                logger.debug("Callback case linear")
                mono_signal_chunk, self._theta_last, self._omega_last =\
                    self.calculate_lin_sweep(t_array,
                                             self._theta_last,
                                             self._omega_last,
                                             target_omega,
                                             )

            # There was a omega=0 case. Reset the theta and omega values.
            if target_omega == 0:
                self._bring_sweep_states_to_zero(self.stream.channels)

            # set the signals to rms = 1 and also do the smooth crossing between voltages
            logger.debug(f"self._sweep_level_last, self._sweep_signal_rms: {self._sweep_level_last}, {self._sweep_signal_rms}")
            mono_signal_chunk = mono_signal_chunk  / np.exp2(-0.5) * make_fade_window_n(self._sweep_level_last,
                                                                                        self._sweep_signal_rms,
                                                                                        frames,
                                                                                        )
            self._sweep_level_last = self._sweep_signal_rms

            # If this was the last fade-out callback and calling back needs to stop
            if self.fade_out_frames["remaining"] <= frames:
                stream_needs_to_stop_now = True

            # Apply fade-out
            if not np.isnan(self.fade_out_frames["remaining"]):
                fade_start_end_idxs = (self.fade_out_frames["remaining"] - self.fade_out_frames["total"],
                                      self.fade_out_frames["remaining"],
                                      )
                fade_out_window = make_fade_window_n(1,
                                                     0,
                                                     frames,
                                                     fade_start_end_idxs,
                                                     )
                mono_signal_chunk = mono_signal_chunk * fade_out_window

                logger.debug(f"Remaining/prepared fade out frames: {self.fade_out_frames['remaining']}/{len(fade_out_window)}")
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
            logger.debug(f"_sweep_channel, _sweep_signal_rms: {self._sweep_channel}, {self._sweep_signal_rms}")
            target_rms_levels[self._sweep_channel - 1] = 1
            # dynamic level adjustment is handled in the fade functions (ramping). therefore here the gain is neutral.
            logger.debug(f"Sweep levels: {target_rms_levels}")

            # Tell Main window which frequency you are at
            if self._sweep_signal_rms == 0 or target_omega == 0:
                self.sweep_generated.emit(np.nan, self.stream.latency)
            else:
                self.sweep_generated.emit(self._omega_last / 2 / np.pi, self.stream.latency)
            # doesn't work on initiation  # what??

            return mono_signal_chunk, initial_rms, target_rms_levels, stream_needs_to_stop_now

        except Exception as e:
            logger.critical(
                f"Failed to add {frames} frames during sweep generator callback. Error: {repr(e)}")
            raise sd.CallbackAbort

    def callback(self, indata, frames, ctime, status):
        """
        Callback function for sounddevice player.
        Initiated wheneversound device runs out of buffer.
        Avoid placing memory allocation or i/o tasks in here.
        """
        logger.debug("")
        logger.debug(f"----Callback for DAC time: {ctime.outputBufferDacTime}----")
        if logger.level < 20:  # 20 is info, 10 is debug
            t1_start = time.perf_counter_ns()

        if status.output_underflow:
            self.log_through_thread.emit("Buffer underflow. Consider increasing latency settings.")
            # raise sd.CallbackAbort
            # Maybe switch to high latency if this occurs
        elif status and not status.priming_output:
            error_message = f"Unexpected callback status: {status}"
            logger.warning(error_message)

        # Nothing to play
        if (self.play_pos is None) and (np.isnan(self.user_req_alpha) and np.isnan(self.user_req_omega)):
            mono_signal_chunk = np.zeros(frames)          
            logger.debug("Nothing to play for the callback. Put in zeros.")

        # Play a user generated signal
        elif self.play_pos is not None:
            mono_signal_chunk, initial_rms, target_rms_levels, stream_needs_to_stop_now = self.callback_for_ugs(frames)

        # Play a sweep
        elif (not np.isnan(self.user_req_alpha)) or (not np.isnan(self.user_req_omega)):
            mono_signal_chunk, initial_rms, target_rms_levels, stream_needs_to_stop_now = self.callback_for_sweep(frames)

        # Write to sound card
        indata[:frames, :self.stream.channels] = mono_signal_chunk\
            .repeat(self.stream.channels, axis=0)\
            .reshape(frames, self.stream.channels)\
            / initial_rms * np.array(target_rms_levels)  # scale for correct voltages

        # log the output signal
        if self.log_output_signal:
            logger.info(f"Adding to log the signal. Remaining fade frames after this: {self.fade_out_frames['remaining']}")
            self.output_log["time_sig"].extend([float(i) for i in indata[:, 0]])  # only channel 1 is logged

            max_length = int(self.stream.samplerate * 3)
            if len(self.output_log["time_sig"]) > max_length:
                self.output_log["time_sig"] = self.output_log["time_sig"][-max_length:]

            if len(self.output_log["fade_out_window"]) > max_length:
                self.output_log["fade_out_window"] = self.output_log["fade_out_window"][-max_length:]

            if len(self.output_log["fade_in_window"]) > max_length:
                self.output_log["fade_in_window"] = self.output_log["fade_in_window"][-max_length:]

            logger.debug(f"Callback current / buffer DAC time: {ctime.currentTime} / {ctime.outputBufferDacTime}")
            if logger.level < 20:  # 20 is info, 10 is debug
                logger.debug(f"Calculation / play time: {(time.perf_counter_ns() - t1_start) / 1e6:.3f} ms / {frames / self.stream.samplerate * 1000:.3f} ms")

        # Playing needs to stop
        if stream_needs_to_stop_now:
            raise sd.CallbackStop()


    @qtc.Slot(dict)
    def sweep_play(self, **kwargs):
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

            # If no stream yet, no active sweep stream or an ongoing ugs stream
            if self.stream is None or not self.stream.active or self.play_pos:
                self.initiate_stream()
                
            if self.stream is not None:    
                self.stream.start()
                logger.info("Sweep stream started.")
            
            else:
                self.signal_exception.emit("Stream not available. Sweep could not start.")

        except Exception as e:
            self.signal_exception.emit(str(e))
            logger.critical(f"Sweep generator failed. {e}")
            if self.stream is not None:
                self.stream.close(ignore_errors=True)

    @qtc.Slot(dict, dict)
    def ugs_play(self, play_kwargs):
        "UGS means 'user generated signal'"
        try:
            # Make sure stream is stopped first
            self.stop_play_blocking()
            
            # Initiate a stream
            if self.stream is not None and self.stream.samplerate == play_kwargs["signal_object"].FS:
                self.initiate_stream()
            else:
                self.initiate_stream(force_sample_rate=play_kwargs["signal_object"].FS)
            
            if self.stream is None:    
                self.signal_exception.emit("Stream not available. Play could not start.")
                return

            self.user_gen_signal = play_kwargs["signal_object"]
            self.set_ugs_play_levels(play_kwargs["requested_voltages"])

            self.is_play_in_loop = play_kwargs["loop"]
            self.stop_after_seconds = play_kwargs["stop_after_seconds"]
            self.play_pos = 0

            status_info_text = "---- Playing ----" if not play_kwargs["loop"] else "---- Playing in loop ----"
            now = time.time()
            status_info_text += f"\nLocal time at start: {datetime.fromtimestamp(now).strftime('%B %d, %H:%M:%S')}"
            stop_after_seconds = play_kwargs.get("stop_after_seconds", 0)
            if stop_after_seconds > 0:
                stop_time = now + play_kwargs["stop_after_seconds"]
                status_info_text += f"\nLocal time to stop: {datetime.fromtimestamp(stop_time).strftime('%B %d, %H:%M:%S')}"

            for cn in range(1, self.stream.channels + 1):
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

                    
            # ---- Stop timer
            if self.stop_after_seconds > 0:
                self.stop_timer = BasicCountDownTimer(self.stop_after_seconds)
                self.stop_timer.signal_finished.connect(self.stop_play)
                self.play_stopped.connect(self.stop_timer.stop)
                qtw.QApplication.instance().aboutToQuit.connect(self.stop_timer.stop)
                self.stop_timer.start()

            self.stream.start()
            self.play_started.emit(status_info_text)

            logger.debug(f"Stream started with block sizes {self.stream.blocksize}.")

        except Exception as e:
            self.signal_exception.emit(repr(e))
            logger.error(f"Play_once failed during start. {e}")
            if self.stream is not None:
                self.stream.close(ignore_errors=True)

    @qtc.Slot(str)
    def stop_play(self):
        if self.stream is not None and self.stream.active:
            if np.isnan(self.fade_out_frames["remaining"]):
                self.fade_out_frames = {"remaining": self.fade_window_size,
                                        "total": self.fade_window_size,
                                        "stop_after": True,
                                        }
            else:
                pass
            logger.debug("Stop fadeout initiated.")

        else:
            logger.debug("Stream was not active when stop was requested.")
    
    def stop_play_blocking(self):
        self.stop_play()
        # Block until
        while self.stream is not None and self.stream.active:
            pass

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
        logger.debug("User generated signal play levels updated in player.")

    @qtc.Slot(float)
    def set_sweep_level(self, voltage: float) -> None:
        """
        Receive requested voltage output and set the correct gain in a dictionary attribute.
        """
        rms_required = self.calculate_digital_signal_rms({self._sweep_channel: voltage}, np.exp2(0.5))[self._sweep_channel]
        self._sweep_voltage = float(voltage)

        if not hasattr(self, "_sweep_signal_rms"):
            self._sweep_signal_rms = 0.
        self._sweep_signal_rms = rms_required

        logger.debug("Sweep level updated in player.")

        # not handling exceptions within this function. risky.

    @qtc.Slot(int)
    def set_sweep_channel(self, channel):
        """Sets channel for sweep as an integer value.
        The integer for channel is user friendly, starting from 1.
        """
        self.stop_play_blocking()  # for user hearing safety

        self._sweep_channel = int(channel)
        self.set_sweep_level(self._sweep_voltage)

        logger.debug("Sweep channel updated in player.")


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
        logger.debug(f"Writer thread started with params: {self.generated_signal.analysis}, {self.kwargs}")
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
                logger.info(f"File '{self.kwargs['file_name']}' write successful.")
        except Exception as e:
            self.file_write_fail.emit("Error during file write: " + str(e))
            raise e


class PlayerLogger(qtc.QThread):
    # Why do I need this???
    def __init__(self):
        super().__init__()
        # self.setPriority(qtc.QThread.LowestPriority)

    @qtc.Slot()
    def log(self, message):
        logger.info(message)


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
    play_sample_rate: int = 48000
    stream_latency: str = "high"
    file_folder: str = ""

    def __post_init__(self):
        settings_storage_title = (self.app_name
                                  + " v"
                                  + (".".join(self.version.split(".")[:2])
                                     if "." in self.version
                                     else "???"
                                     )
                                  )
        self.settings_sys = qtc.QSettings(
            self.author_short, settings_storage_title)
        self.read_all_from_registry()
        self._field_types = {field.name: field.type for field in fields(self)}
        
    def update(self, attr_name, new_val):
        # Update a given setting
        # Check type of new_val first
        expected_type = self._field_types[attr_name]
        if type(new_val) != expected_type:
            raise TypeError(f"Incorrect data type received for setting '{attr_name}'. Expected type: {expected_type}. Received type/value: {type(new_val)}/{new_val}.")
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
            settings[field.name] = getattr(self, field.name)
        return settings

    def __repr__(self):
        return str(self.as_dict())


class CountDownTimer(qtc.QTimer):
    """
    Timer object that sends an update of remaining time at every 'interval' time, in ms.
    When 'total_duration' is reached, the object sends a 'signal_finished' with argument 'total_duration', in s.
    Use inherited 'start' method to start it.
    """
    signal_intermediate_update = qtc.Signal(float)  # return how many seconds are left
    signal_finished = qtc.Signal(float)  # return once again the original total duration in seconds

    def __init__(self, update_interval, total_duration):
        super().__init__()
        self.update_interval = update_interval  # in ms
        self.total_duration = total_duration  # in s
        self.elapsed = 0.  # in s
        self.setInterval(update_interval)

        self.timeout.connect(self.update_elapsed_time)
        self.timeout.connect(self.signal_out_remaining_time)
        self.timeout.connect(self.check_if_its_time_to_finish)
    
    def update_elapsed_time(self):
        self.elapsed += self.update_interval / 1000
        # this isn't accurate it seems...
    
    def calc_remaining_time(self):
        return self.total_duration - self.elapsed

    def signal_out_remaining_time(self):
        self.signal_intermediate_update.emit(self.calc_remaining_time)
    
    def check_if_its_time_to_finish(self):
        if self.elapsed >= self.total_duration:
            self.signal_finished.emit(self.total_duration)
            self.stop()


class BasicCountDownTimer(qtc.QTimer):
    """
    Timer object that sends a signal when 'total_duration' is reached.
    The object sends a 'signal_finished' with argument 'total_duration', in s.
    Use inherited 'start' method to start it.
    """
    signal_finished = qtc.Signal()

    def __init__(self, total_duration: (int, float)):
        super().__init__()
        self.setInterval(total_duration * 1000)
        self.setSingleShot(True)

        self.timeout.connect(self.signal_finished)


class MainWindow(qtw.QMainWindow):
    global settings, app_definitions

    gen_signal_not_ready = qtc.Signal(str)
    gen_parameters_changed = qtc.Signal()
    play_parameters_changed = qtc.Signal()
    sys_parameters_changed = qtc.Signal()
    
    request_generator_generate_ugs = qtc.Signal(str, dict)
    request_generator_process_imported_file = qtc.Signal(str, dict)
    request_generator_import_file = qtc.Signal(str)
    request_generator_clear_imported_file = qtc.Signal()
    
    update_signal_info_widget = qtc.Signal(str)
    update_play_info_widget = qtc.Signal(str)
    
            
    # ---- Start player and generator

    def setup_player_thread(self):
        self.player_logger = PlayerLogger()
        self.player_logger.start(qtc.QThread.LowestPriority)
        
        qtw.QApplication.instance().aboutToQuit.connect(self.player_logger.quit)

        self.player_thread = qtc.QThread()
        self.player = Player()
        self.player.moveToThread(self.player_thread)

    def setup_generator_thread(self):
        self.generator = Generator()
        self.generator_thread = qtc.QThread()
        self.generator.moveToThread(self.generator_thread)

    def setup_poll_sound_devices_thread(self): 
        # Update detected sound devices every N seconds
        self.poll_sound_devices_timer = qtc.QTimer()
        self.poll_sound_devices_timer.setInterval(2000)
        self.poll_sound_devices_timer.start()  # priority tanımlayınca interval devre dışı kaldı. garip.

        self.poll_sound_devices_timer.timeout.connect(self.player.poll_sound_devices)
        self.player.signal_sound_devices_polled.connect(self.sound_device_info_widget.setText)
    
    @qtc.Slot(TestSignal)
    def handler_generator_signal_ready(self, generated_signal, power_spectrum, octave_bands):
        logger.debug("Main window received signal 'Generated signal ready'")
        try:
            self.generated_signal = generated_signal

            # Update user with the changes
            self.update_signal_info_widget.emit("Signal generated. Analyzing...")                
            self.mpl_widget.update_plot(self.generated_signal, power_spectrum, octave_bands)
            generator_info_text = self.generated_signal.analysis

        except Exception as e:
            self.gen_signal_not_ready.emit(
                "Failed to receive generated signal from generator thread.\n" + str(e))
        else:  # do this always
            self.update_signal_info_widget.emit(generator_info_text)
            
        self.generate_group.setEnabled(True)

    @qtc.Slot(str)
    def handler_generator_signal_not_ready(self, generator_info_text):
        "Signal not ready"
        self.mpl_widget.clear_plot()
        self.player.stop_play()
        self.update_signal_info_widget.emit(generator_info_text)
        self.generated_signal = None
        
    @qtc.Slot(TestSignal)
    def handle_generator_file_imported_with_success(self, imported_signal):
        # A file is imported into the generator successfully
        index_to_set = self.sample_rate_selector.findData(imported_signal.FS)
        if index_to_set == -1:
            self.sample_rate_selector.addItem(str(imported_signal.FS), imported_signal.FS)
        else:
            self.sample_rate_selector.setCurrentIndex(index_to_set)
        self.duration_widget.setValue(imported_signal.T)
        self.gen_signal_not_ready.emit((f"Imported successfully.\n{imported_signal.initial_data_analysis}"
                                        "\n\nSet your filters and press 'Generate' to continue."
                                        )
                                       )

    @qtc.Slot(Exception)
    def handler_generator_exception(self, e):
        error_text = "Error in signal generator."
        informative_text = str(e)
        self.handler_generator_signal_not_ready(error_text)
        PopupError(error_text, informative_text=informative_text)
        self.generate_group.setEnabled(True)
        

    def make_connections_and_start_threads(self):
        self.update_signal_info_widget.connect(self.signal_info_widget.setText)
        self.update_play_info_widget.connect(self.play_info_widget.setPlainText)
        self.player.signal_exception.connect(self.player_exception)
        self.player.play_stopped.connect(self.play_stopped)
        
        # Generator signals
        self.gen_signal_not_ready.connect(self.handler_generator_signal_not_ready)
        self.generator.signal_ready.connect(self.handler_generator_signal_ready)
        self.generator.signal_not_ready.connect(self.handler_generator_signal_not_ready)
        self.generator.file_import_success.connect(self.handle_generator_file_imported_with_success)
        self.generator.exception.connect(self.handler_generator_exception)
        
        # Generator slots
        self.request_generator_generate_ugs.connect(self.generator.generate_ugs)
        self.request_generator_process_imported_file.connect(self.generator.process_imported_file)
        self.request_generator_import_file.connect(self.generator.import_file)
        self.request_generator_clear_imported_file.connect(self.generator.clear_imported_file)

        # Cleaning threads on exit
        qtw.QApplication.instance().aboutToQuit.connect(self.player_thread.quit)
        qtw.QApplication.instance().aboutToQuit.connect(self.generator_thread.quit)
        
        # Start threads
        self.generator_thread.start(qtc.QThread.LowPriority)
        logging.debug(f"Generator thread id: {self.generator_thread.currentThread()}")
        
        self.player_thread.start(qtc.QThread.TimeCriticalPriority)
        logging.debug(f"Player thread id: {self.player_thread.currentThread()}")
        
        self.setup_poll_sound_devices_thread()

    def __init__(self, app):  # is this app thing really necessary?
        """MainWindow constructor"""
        super().__init__()
        
        self.setup_generator_thread()
        self.setup_player_thread()
    
        # Main UI code goes here
        self.setMinimumWidth(1024)
        self.setWindowTitle(" - ".join(
            (app_definitions["app_name"],
             app_definitions["version"])
            ))

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

        self.duration_widget = qtw.QDoubleSpinBox(Minimum=1,
                                             Maximum=60*10,
                                             Value=60,
                                             Decimals=2,
                                             ToolTip="Duration of generated signal in seconds."
                                                     "\nMaximum allowed value is 600."
                                                     "\nWarning: Long signals with high sampling rates "
                                                     "will take a long time to generate!"
                                             )
        self.duration_widget.valueChanged.connect(self.gen_parameters_changed)

        self.sample_rate_selector = qtw.QComboBox()

        sample_rate_list = [22050, 44100, 48000, 96000]
        for i in sample_rate_list:
            self.sample_rate_selector.addItem(str(i), i)
            
        try:
            self.sample_rate_selector.setCurrentIndex(sample_rate_list.index(settings.play_sample_rate))
        except ValueError:
            self.sample_rate_selector.setCurrentIndex(1)

        self.sample_rate_selector.currentTextChanged.connect(self.gen_parameters_changed)

        # Filters
        self.no_of_filters = 8

        class Filter():
            def __init__(self, parent):
                super().__init__()
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

                for widget in self.widgets.values():
                    self.layout.addWidget(widget)
            
            def as_dict(self):
                return {"type": self.widgets["type"].currentText(),
                        "frequency": self.widgets["frequency"].value(),
                        "order": self.widgets["order"].currentData(),
                    }

        filters = [Filter(parent=self) for i in range(self.no_of_filters)]
        
        # add a basic HP filter to avoid DC offset
        filters[0].widgets["type"].setCurrentText("HP")
        filters[0].widgets["frequency"].setValue(1)

        # Generator parameters form
        gen_form_layout = qtw.QFormLayout()
        gen_form_layout.addRow("Signal type", signal_type_selector)
        gen_form_layout.addRow("Frequency", frequency_widget)
        gen_form_layout.addRow(pwi.SunkenLine())
        for i in range(self.no_of_filters):
            gen_form_layout.addRow(f"Filter {i + 1}", filters[i].layout)
        gen_form_layout.addRow(pwi.SunkenLine())
        gen_form_layout.addRow("Compression", compression_widget)
        gen_form_layout.addRow(pwi.SunkenLine())
        gen_form_layout.addRow("Duration", self.duration_widget)
        gen_form_layout.addRow("Sample rate", self.sample_rate_selector)
        gen_form_layout.addRow(pwi.SunkenLine())

        # 'Generate' button
        generate_button = qtw.QPushButton("Generate",
                                          MinimumHeight=40,
                                          )

        # Make the total layout and widget of generator group
        self.generate_group = qtw.QWidget()
        self.generate_group.setLayout(qtw.QVBoxLayout())

        # Add the widgets, layouts
        self.generate_group.layout().addLayout(gen_form_layout)
        # self.generate_group.layout().addSpacing(10)
        self.generate_group.layout().addWidget(generate_button)

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

        stop_after_widget = qtw.QDoubleSpinBox(Minimum=0,
                                               Maximum=1e6-1,
                                                Value=0,
                                                Decimals=1,
                                                SingleStep=30,
                                                ToolTip=("Stop the playback after the user defined period of time"
                                                         " is passed. Value is in minutes. '0' means disabled."
                                                         ),
                                                )
        stop_after_widget.valueChanged.connect(self.play_parameters_changed)

        # Player parameters form
        self.player_params_widget = qtw.QWidget()
        play_params_form_layout = qtw.QFormLayout()
        play_params_form_layout.setContentsMargins(0, 0, 0, 0)
        self.player_params_widget.setLayout(play_params_form_layout)

        play_params_form_layout.addWidget(sys_gain_adjust_button)
        play_params_form_layout.addWidget(qtw.QFrame(FrameShape=qtw.QFrame.HLine,
                                                     FrameShadow=qtw.QFrame.Sunken),
                                          )

        for i in level_widgets.keys():
            play_params_form_layout.addRow(f"Output voltage for Ch. {i}", level_widgets[i])
        play_params_form_layout.addRow(pwi.SunkenLine())
        play_params_form_layout.addRow("Play in loop", play_in_loop_widget)
        play_params_form_layout.addRow("Stop after (minutes)", stop_after_widget)
        play_params_form_layout.addRow(pwi.SunkenLine())
        play_params_form_layout.addRow("Speaker nominal impedance", speaker_nominal_impedance_widget)
        play_params_form_layout.addRow("Nominal power at speaker", speaker_nominal_power_widget)

        # Buttons
        self.play_button = qtw.QPushButton("Play",
                                      MinimumHeight=40,
                                      )
        stop_button = qtw.QPushButton("Stop",
                                      MinimumHeight=40,
                                      )
        player_buttons_layout = qtw.QHBoxLayout()
        player_buttons_layout.addWidget(self.play_button)
        player_buttons_layout.addWidget(stop_button)

        # Sound device info
        self.sound_device_info_widget = qtw.QTextEdit(readOnly=True)

        # Make the total layout and widget of generator group
        player_group = qtw.QWidget()
        play_group_layout = qtw.QVBoxLayout()
        player_group.setLayout(play_group_layout)

        # Add the widgets, layouts
        play_group_layout.addWidget(self.player_params_widget)
        play_group_layout.addWidget(pwi.SunkenLine())
        play_group_layout.addSpacing(10)
        play_group_layout.addWidget(qtw.QLabel("<b>Sound Device Information</b>"),
                                    alignment=qtc.Qt.AlignHCenter,
                                    )
        play_group_layout.addWidget(self.sound_device_info_widget)
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
        voltage_spin_box.setValue(1)

        # voltage_spin_box.lineEdit().setReadOnly(True)  # for safety during development
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
        file_rms_title = qtw.QLabel("<b>File RMS level</b>",
                                    alignment=qtc.Qt.AlignHCenter)
        file_rms_description = qtw.QLabel("Exported signal RMS = 20*log10(A) + B dBFS",
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
        write_file_form_layout.addWidget(file_rms_description)
        write_file_form_layout.addRow("A (multiplier)", self.file_rms_multiplier_widget)
        write_file_form_layout.addRow("B (dB)", self.file_rms_db_widget)
        write_file_form_layout.addRow("File format", self.file_format_widget)
        write_file_form_layout.addRow(pwi.SunkenLine())

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
        self.signal_info_widget = qtw.QTextEdit(readOnly=True, parent=self)
        self.signal_info_widget.setSizePolicy(qtw.QSizePolicy.MinimumExpanding,
                                                   qtw.QSizePolicy.Preferred,
                                                   )
        # Is there a way to set the size policies with constructor arguments?

        self.play_info_widget = qtw.QTextEdit(readOnly=True)
        self.play_info_widget.setSizePolicy(qtw.QSizePolicy.MinimumExpanding,
                                         qtw.QSizePolicy.Preferred,
                                         )

        # ---- About tab
        about_group = qtw.QLabel(alignment=qtc.Qt.AlignCenter)

        about_text = "\n".join([
        f"{app_definitions['description']}",
        f"Version: {app_definitions['version']}",
        "",
        f"{app_definitions['copyright']}",
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
        mw_left_widget.addTab(self.generate_group, "Generator")
        mw_left_widget.addTab(player_group, "Player")
        mw_left_widget.addTab(write_file_group, "Write file")
        mw_left_widget.addTab(sweep_group, "Sweep generator")
        mw_left_widget.addTab(about_group, "About")
        mw_left_widget.setMinimumWidth(500)
        mw_left_widget.setSizePolicy(qtw.QSizePolicy.Minimum, qtw.QSizePolicy.Minimum)

        # ---- Layout Right side (generated signal data)
        mw_right_widget = qtw.QWidget()
        mw_right_layout = qtw.QVBoxLayout(mw_right_widget)
        # mw_right_widget.setLayout(mw_right_layout) already given in above line to layout

        mw_right_layout.addWidget(qtw.QLabel("<b>Generated Signal Information</b>"),
                                  alignment=qtc.Qt.AlignHCenter,
                                  )
        mw_right_layout.addWidget(self.signal_info_widget)

        self.mpl_widget = MatplotlibWidget(self)
        self.mpl_widget.setMinimumWidth(400)
        self.mpl_widget.canvas.setSizePolicy(qtw.QSizePolicy.MinimumExpanding,
                                        qtw.QSizePolicy.Expanding,
                                        )
        mw_right_layout.addWidget(self.mpl_widget, 3)
        mw_right_layout.addWidget(qtw.QFrame(FrameShape=qtw.QFrame.HLine,
                                             FrameShadow=qtw.QFrame.Sunken),
                                  )

        mw_right_layout.addWidget(qtw.QLabel("<b>Player status</b>"),
                                  alignment=qtc.Qt.AlignHCenter,
                                  )
        mw_right_layout.addWidget(self.play_info_widget, 1)

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


        # ---- Functions triggered by user through the GUI
        def gain_and_levels_button_clicked():
            self.player.stop_play()
            self.player.poll_sound_devices()
            sys_gain_widget = SysGainAndLevelsPopup()
            sys_gain_widget.user_changed_sys_params_signal.connect(self.sys_parameters_changed)
            sys_gain_widget.exec()

        def play_clicked():
            if not hasattr(self, "generated_signal") or not isinstance(self.generated_signal, TestSignal):
                error_text = "No signal found to play."
                informative_text = "Generate a signal using the generator tab."
                PopupError(error_text, informative_text=informative_text)
                return

            else:
                # Params to play signal
                requested_voltages = \
                    {n_c: level_widgets[n_c].value() for n_c in range(1, settings.channel_count + 1)}

                play_kwargs = {
                    "signal_object": self.generated_signal,
                    "loop": play_in_loop_widget.checkState(),
                    "requested_voltages": requested_voltages,
                    "stop_after_seconds": stop_after_widget.value() * 60,
                    }

                self.player.ugs_play(play_kwargs)

        def generate_clicked():
            try:
                # Make the signal
                sig_type = signal_type_selector.currentText()
                kwargs = {"filters": [filter.as_dict() for filter in filters],
                          "frequency": frequency_widget.value(),
                          "compression": compression_widget.value(),
                          "T": self.duration_widget.value(),
                          "FS": self.sample_rate_selector.currentData(),
                          }
                self.generate_group.setEnabled(False)
                if sig_type == "Imported":
                    self.request_generator_process_imported_file.emit("Reuse existing", kwargs)
                else:
                    self.request_generator_generate_ugs.emit(sig_type, kwargs)
                    
            except Exception as e:
                error_text = "Unable to place generator request in the generator thread."
                logger.critical(str(e))
                PopupError(error_text, informative_text=str(e))
                self.generate_group.setEnabled(True)

        def write_file_clicked():
            if not self.generated_signal:
                error_text = "No signal found to write."
                informative_text = "Generate a signal using the generator tab."
                PopupError(error_text, informative_text=informative_text)
                return
            write_args = {"file_format": self.file_format_widget.currentText(),
                          "file_rms": 10**(self.file_rms_db_widget.value() / 20) * self.file_rms_multiplier_widget.value(),
                          }
            if write_args["file_rms"] * self.generated_signal.CF > 1:
                error_text = "Current settings will cause digital clipping."
                informative_text = "Reduce target RMS voltage and/or signal crest factor.\nMake sure system gain is entered correctly and increase amplifier gain if necessary."
                PopupError(error_text, informative_text=informative_text)
                return

            self.player.stop_play()
            file_filters = {"FLAC": "FLAC files (*.flac)",
                            "WAV": "Wave files (*.wav)",
                            "OGG": "Vorbis files (*.ogg)",
                            }
            
            # add functionality for remembering latest file_folder
            file_folder = Path(settings.file_folder)
            if not file_folder.is_dir():
                file_folder = Path.cwd()
            
            path_unverified = qtw.QFileDialog.getSaveFileName(None, "Save audio signal in file...",
                                                                      str(file_folder),
                                                                      file_filters[write_args["file_format"]],
                                                                      "",
                                                                      )
            
            try:
                file_raw = path_unverified[0]
                if file_raw:
                    file = Path(file_raw)
                    assert file.parent.exists()
                    settings.update("file_folder", str(file.parent))
                    # is this app thing really necessary? why not use qtw.QApplication.instance()
                    # in Nautilus the filenames come without suffixes. therefore added below line.
                    write_args["file_name"] = Path(str(file) + "." + write_args["file_format"].lower() if str(file)[-3:].lower() != write_args["file_format"][-3:].lower() else str(file))
                    writer = FileWriter(app, self.generated_signal, **write_args)
                    writer.file_write_successful.connect(write_file_info_widget.setText)
                    writer.file_write_busy.connect(write_file_info_widget.setText)
                    writer.file_write_fail.connect(write_file_info_widget.setText)
                    writer.finished.connect(lambda: logger.debug("Finished thread file writer"))
                    writer.start()
                else:
                    logger.debug("Save file selection canceled or invalid save file.")
                    write_file_info_widget.setText("Invalid or empty save file.")
                
            except Exception as e:
                error_text = "File writer failed."
                PopupError(error_text, informative_text=str(e))

        def choose_import_file():
            try:
                self.player.stop_play()

                # Wait if playback is going on. This call does file access and can cause
                # buffer underrun in player callback
                for timer in range(10):
                    if self.player.stream.active:
                        qtc.QThread.msleep(100)
                    if timer == 99:
                        raise RuntimeError("Could not stop player thread.")
                    else:
                        qtc.QThread.msleep(100)
                        break

                # add functionality for remembering latest file_folder
                file_folder = Path(settings.file_folder)
                if not file_folder.is_dir():
                    file_folder = Path.cwd()

                # ask user to pick file
                file_formats = " ".join(["*." + str(suffix).lower() for suffix in sf.available_formats()])
                file_raw = qtw.QFileDialog.getOpenFileName(None,
                                                            "Choose audio file to import...",
                                                            str(file_folder),
                                                            f"Audio files ({file_formats})",
                                                            )[0]
                if file_raw and (file := Path(file_raw)).is_file():
                    settings.update("file_folder", str(file.parent))
                    self.request_generator_import_file.emit(str(file))
                else:
                    self.gen_signal_not_ready.emit("No file chosen.")
                    self.request_generator_clear_imported_file.emit()

            except Exception as e:
                error_text = "File import failed."
                PopupError(error_text, informative_text=str(e))
                settings.update("file_folder", "")

        def request_sweep(dial_value):
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
                self.player.sweep_play(target_freq=freq_on_dial)

            except Exception as e:
                error_text = "Unable to place sweep generate request in the player thread."
                logger.critical(repr(e))
                PopupError(error_text, informative_text=repr(e))

        # User changed generator signal type
        def signal_type_selection_changed():
            self.duration_widget.setEnabled(signal_type_selector.currentText() != "Imported")

            if signal_type_selector.currentText() == "Imported":
                choose_import_file()
        signal_type_selector.activated.connect(signal_type_selection_changed)

        # Disabling voltage widgets for disabled channels
        def update_gui_for_change_in_number_of_channels():
            sweep_channel.setMaximum(int(settings.channel_count))
            for i, level_widget in level_widgets.items():
                level_widget.setEnabled(i <= int(settings.channel_count))
        update_gui_for_change_in_number_of_channels()

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
        freq_dial.valueChanged.connect(request_sweep)

        sweep_channel.valueChanged.connect(self.player.set_sweep_channel, qtc.Qt.QueuedConnection)
        sweep_channel.valueChanged.emit(sweep_channel.value())

        voltage_spin_box.valueChanged.connect(self.player.set_sweep_level, qtc.Qt.QueuedConnection)
        voltage_spin_box.valueChanged.emit(voltage_spin_box.value())

        self.player.sweep_generated.connect(freq_display.display, qtc.Qt.QueuedConnection)

        def unavailable_feature():
            error_text = "Feature not implemented yet."
            PopupError(error_text)

        # ---- Connection of pushbuttons
        self.play_button.clicked.connect(play_clicked)
        stop_button.clicked.connect(self.player.stop_play)
        sweep_stop_button.clicked.connect(self.player.stop_play)
        generate_button.clicked.connect(generate_clicked)
        sys_gain_adjust_button.clicked.connect(gain_and_levels_button_clicked)
        sys_gain_adjust_button_2.clicked.connect(gain_and_levels_button_clicked)
        write_file_button.clicked.connect(write_file_clicked)

        # ---- Functions triggered by threads and logic, not the user
        
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
        def player_log_through_thread(message):
            self.player_logger.log(f"Player: {message}")
        self.player.log_through_thread.connect(player_log_through_thread)

        # Logging functionality
        def show_log(log_dict):
            log_win = LogView(log_dict)
            log_win.exec()
        self.player.publish_log.connect(show_log)

        # Output voltage request not feasible
        def impossible_voltage_request_happened_at_sweeper(str):
            voltage_spin_box.setValue(0)  # not very user friendly
        self.player.impossible_voltage_request.connect(impossible_voltage_request_happened_at_sweeper)

        @qtc.Slot(str)
        def play_started(play_info_text):
            "User generated signal play started"
            self.player_params_widget.setEnabled(False)
            self.play_button.setEnabled(False)
            # stop_button.setEnabled(True)
            self.update_play_info_widget.emit(play_info_text)
            
        self.player.play_started.connect(play_started)

        @qtc.Slot(str)
        def gen_parameters_changed(new_param="//"):
            "When generator parameters changed"
            if (signal_type_selector.currentText() != "Imported" or
                    (hasattr(self, "generated_signal") and isinstance(self.generated_signal, TestSignal))):
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
            update_gui_for_change_in_number_of_channels()
            self.player.set_sweep_level(voltage_spin_box.value())  # due to a bug where the levels are not updated
            # after system gain settings are changed by user
            # setting the maximum value for the sweep voltage spin box here would be nice
            # but it depends on channel so not so simple to do

        self.sys_parameters_changed.connect(sys_parameters_changed_actions)
        
        self.make_connections_and_start_threads()
        
    @qtc.Slot(Exception)
    def player_exception(self, e):
        error_text = "Error in player."
        informative_text = str(e)
        PopupError(error_text, informative_text=informative_text, parent=self)
        self.play_stopped("Stopped due to error in player.")

    @qtc.Slot(str)
    def play_stopped(self, stop_info_text):
        "User generated signal play stopped"
        self.player_params_widget.setEnabled(True)
        self.play_button.setEnabled(True)
        # stop_button.setEnabled(False)
        self.update_play_info_widget.emit(stop_info_text)


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

    @qtc.Slot(TestSignal)
    def update_plot(self, generated_signal, power_spectrum, octave_bands):
        self.ax.cla()
        if generated_signal:
            # # Power spectrum of signal
            # FS = generated_signal.FS
            # PowerSpect = signal.welch(generated_signal.time_sig.astype("float32"),
            #                           fs=FS,
            #                           nperseg=FS/4,  # defines also window size
            #                           window="hann",
            #                           scaling="spectrum")

            # # Power per octave band of signal
            # center_frequencies, three_oct_power = calculate_3rd_octave_bands(generated_signal.time_sig, FS, multiprocess=False)

            FS = generated_signal.FS
            self.ax.semilogx(*power_spectrum, label="Power spectral density")
            self.ax.step(*octave_bands, where="mid", label="1/3 octave bands")

            self.ax.set_xlim(10, FS/2)
            self.ax.set_ylim(-70, 5)
            self.ax.grid(which='minor', axis='x')
            self.ax.grid(which='major', axis='y')
            self.ax.legend()

        self.canvas.draw()

    def clear_plot(self):
        self.update_plot(None, None, None)


def get_main_dir():
    
    if getattr(sys, 'frozen', False):
        # The application is frozen
        return Path(sys.executable).parent
        
    else:
        # The application is not frozen
        return Path(__file__).parent


def parse_args(app_definitions):
    import argparse

    description = (
        f"{app_definitions['app_name']} - {app_definitions['copyright']}"
        "\nThis program comes with ABSOLUTELY NO WARRANTY"
        "\nThis is free software, and you are welcome to redistribute it"
        "\nunder certain conditions. See LICENSE file for more details."
    )

    parser = argparse.ArgumentParser(prog="python main.py",
                                     description=description,
                                     epilog={app_definitions['website']},
                                     )
    parser.add_argument('-d', '--loglevel', nargs="?",
                        choices=["debug", "info", "warning", "error", "critical"],
                        help="Set logging level for Python logging. Valid values are debug, info, warning, error and critical.")

    return parser.parse_args()


def setup_logging(level: str="warning", args=None):
    if args and args.loglevel:
        log_level = getattr(logging, args.loglevel.upper())
    else:
        log_level = level.upper()
        
    log_filename = Path.home().joinpath(f".{app_definitions['app_name'].lower()}.log")
    
    file_handler = logging.FileHandler(filename=log_filename)
    stdout_handler = logging.StreamHandler(stream=sys.stdout)
    handlers = [file_handler, stdout_handler]
    
    logging.basicConfig(handlers=handlers,
                        level=log_level,
                        format="%(asctime)s %(levelname)s - %(funcName)s: %(message)s",
                        force=True,
                        )
    # had to force this
    # https://stackoverflow.com/questions/30861524/logging-basicconfig-not-creating-log-file-when-i-run-in-pycharm
    logger = logging.getLogger()
    logger.info(f"{time.strftime('%c')} - Started logging with log level {log_level}.")
    
    return logger


def main():
    global settings, app_definition, logger

    args = parse_args(app_definitions)
    logger = setup_logging(args=args)
    settings = Settings(app_definitions["app_name"])

    # ---- Create QApplication
    if not (app := qtw.QApplication.instance()):
        app = qtw.QApplication(sys.argv)
        icon_path = str(get_main_dir().joinpath(app_definitions["icon_path"]))
        app.setWindowIcon(qtg.QIcon(icon_path))
    mw = MainWindow(app)

    mw.show()
    app.exec()


if __name__ == "__main__":
    # if os.name == "nt":
    #     # bug: https://stackoverflow.com/questions/22644805/cx-freeze-creates-multiple-instances-of-program
    #     multiprocessing.freeze_support()
    # not using this. buggy still.
    main()
