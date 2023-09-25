import os
import numpy as np
import acoustics as ac  # https://github.com/timmahrt/pyAcoustics
import soundfile as sf


class TestSignal():
    """
    Create a signal object that bears methods to create the
    resulting signal together with its analyses.
    """

    def channel_count(self):
        """Give channel count for a signal in matrix form."""
        try:
            shape = self.time_sig.shape
        except Exception as e:
            raise KeyError("Cannot check the shape of generated time signal.", repr(e))

        if len(shape) == 1:
            return 1
        elif len(shape) > 1:
            return shape[1]
        else:
            raise ValueError("Unrecognized channel count.\n", f"Signal shape: {shape}")

    def __init__(self, sig_type, **kwargs):
        action_for_signal_type = {"Pink noise": "generate_pink_noise",
                                  "White noise": "generate_white_noise",
                                  "IEC 268": "generate_IEC_noise",
                                  "Sine wave": "generate_sine",
                                  "Imported": "import_file",
                                  }
        self.sig_type = sig_type
        self.applied_fade_in_duration = None
        try:
            getattr(self, action_for_signal_type[sig_type])(**kwargs)
            """ runs the correct method to create the time signal.
            (**kwargs) does the running. """
        except KeyError as e:
            raise KeyError("Unrecognized signal type. " + str(e))
        self.apply_processing(**kwargs)

    def reuse_existing(self, **kwargs):
        if kwargs["FS"] != self.FS:
            raise NotImplementedError("Resampling of signal not implemented.")
        self.apply_processing(**kwargs)

    def apply_processing(self, **kwargs):
        if "filters" in kwargs.keys():
            self.apply_filters(**kwargs)
        if "compression" in kwargs.keys():
            self.apply_compression(**kwargs)
        if "set_RMS" in kwargs.keys():
            self.normalize(**kwargs)
        if "fadeinout" in kwargs.keys() and kwargs["fadeinout"]:
            self.apply_fade_in_out()
        self.analyze(**kwargs)  # analyze the time signal

    def generate_pink_noise(self, **kwargs):
        self.make_time_array(**kwargs)
        self.time_sig = ac.generator.pink(len(self.t))

    def generate_white_noise(self, **kwargs):
        self.make_time_array(**kwargs)
        self.time_sig = ac.generator.white(len(self.t))

    def generate_IEC_noise(self, **kwargs):
        self.make_time_array(**kwargs)
        time_sig = ac.generator.pink(len(self.t))
        """
        Do IEC 268 filtering (filter parameters fixed by standard)
        Three first-order high-pass filters at 12.9, 32.4, and 38.5 Hz
        Two first-order low-pass filters at 3900, and 9420 Hz
        """
        time_sig = ac.signal.highpass(time_sig, 12.9, self.FS, order=1)
        time_sig = ac.signal.highpass(time_sig, 32.4, self.FS, order=1)
        time_sig = ac.signal.highpass(time_sig, 38.5, self.FS, order=1)
        time_sig = ac.signal.lowpass(time_sig, 3900, self.FS, order=1)
        time_sig = ac.signal.lowpass(time_sig, 9420, self.FS, order=1)
        self.time_sig = time_sig

    def generate_sine(self, **kwargs):
        self.freq = kwargs["frequency"]
        self.make_time_array(**kwargs)
        self.time_sig = np.sin(self.freq * 2 * np.pi * self.t)

    def import_file(self, **kwargs):
        self.import_file_name = os.path.basename(kwargs["import_file_path"])
        self.time_sig, self.FS = sf.read(kwargs["import_file_path"], always_2d=True)
        self.imported_channel_count = self.channel_count()
        self.imported_FS = self.FS

        if (type(self.FS) is not int) or (self.channel_count() < 1):
            self.time_sig, self.FS = None, None
            raise TypeError("Imported signal is invalid.")

        if "import_channel" in kwargs.keys():
            self.reduce_channels(kwargs["import_channel"])
        else:
            self.reduce_channels("downmix_all")

        self.make_time_array(**kwargs)

        self.raw_import_analysis = (f"File name: {self.import_file_name}"
                                    + f"\nOriginal channel count: {self.imported_channel_count}"
                                    + f"\nImported channel: {str(self.imported_channel + 1) if isinstance(self.imported_channel, int) else self.imported_channel}"
                                    # integer count for user, starting from 1
                                    + f"\nOriginal sample rate: {self.imported_FS}"
                                    )

    def reduce_channels(self, channel_to_use):
        # Channel to use can be an integer starting from 1 or "downmix_all"
        if channel_to_use == 0:
            raise ValueError("Channel numbers start from 1. Channel: 0 is invalid.")
        elif channel_to_use == "downmix_all":
            if self.channel_count() == 1:
                self.time_sig = self.time_sig[:, 0]
                self.imported_channel = 0
                return
            elif self.channel_count() > 1:
                self.time_sig = np.mean(self.time_sig, axis=1)
                self.imported_channel = "downmix_all"
            else:
                raise KeyError(f"Unable to downmix. Channel count {self.channel_count()} is invalid.")

        elif isinstance(channel_to_use, int):
            if channel_to_use > self.channel_count():
                raise KeyError(f"Channel {channel_to_use} does not exist in the original signal.")
            else:
                self.time_sig = self.time_sig[:, channel_to_use - 1]
                self.imported_channel = int(channel_to_use) - 1

        else:
            raise TypeError(f"Invalid request for channel_to_use: {[channel_to_use, type(channel_to_use)]}")

    def analyze(self, **kwargs):
        self.neg_peak = np.min(self.time_sig)
        self.pos_peak = np.max(self.time_sig)
        if -self.neg_peak > self.pos_peak:
            self.peak = self.neg_peak
        else:
            self.peak = self.pos_peak
        self.RMS = ac.signal.rms(self.time_sig)
        self.CF = np.abs(self.peak) / self.RMS
        self.CFdB = 20 * np.log10(self.CF)
        self.mean = np.average(self.time_sig)
        self.analysis = (f"Signal type: {self.sig_type}")

        if self.sig_type == "Imported":
            self.analysis += ("\n" + self.raw_import_analysis)

        self.analysis += (f"\nCrest Factor: {self.CF:.4g}x, {self.CFdB:.2f}dB"
                          + f"\nRMS: {self.RMS:.5g}"
                          + f"\nPositive peak: {self.pos_peak:.5g}"
                          + f"\nNegative peak: {self.neg_peak:.5g}"
                          + f"\nMean: {self.mean:.5g}"
                          + f"\nSample rate: {self.FS} Hz"
                          + f"\nDuration: {self.T:.2f} seconds"
                          + f"\nCurrent channel count: {self.channel_count()}"
                          )

        if self.applied_fade_in_duration:
            self.analysis += (f"\n\nSignal includes fade in/out of " + self.applied_fade_in_duration + ".")

    def apply_compression(self, **kwargs):
        """
        Based on AES standard noise generator, Aug. 9, 2007, Keele
        Only works if input array is normalized to +/-1!!!
        """
        k = 4  # shape factor recommended from the AES tool
        peak_value = np.max(np.abs(self.time_sig))
        a = kwargs.get("compression")

        if a == 0:
            return

        elif a > 0:  # expand
            print("hop")
        # y = sign(x).*exp((log(-1./(exp(log(abs(x + 1e-20))*k+log(a^k/(a^k+1)))-1))+log(abs(x + 1e-20))*k+log(a^k/(a^k+1)))/k)/a;
            self.time_sig /= peak_value
            self.time_sig = np.sign(self.time_sig) * \
                np.exp(
                       (
                           np.log(-1 / (np.exp(np.log(np.abs(self.time_sig + 1e-20)) * k + np.log(a**k/(a**k + 1))) - 1))
                           + np.log(np.abs(self.time_sig + 1e-20)) * k
                           + np.log(a**k/(a**k + 1))
                        )
                       / k
                       ) / a
            self.time_sig *= peak_value

        elif a < 0:  # compress
        # y = sign(x).*(((a*abs(x + 1e-20)).^k./((a*abs(x + 1e-20)).^k + 1)).^(1/k))/((a^k/(a^k + 1))^(1/k));
            self.time_sig /= peak_value
            self.time_sig = np.sign(self.time_sig) * (
                ((a * np.abs(self.time_sig + 1e-20))**k
                 / ((a * np.abs(self.time_sig + 1e-20))**k + 1))**(1 / k)) / ((a**k / (a**k + 1))**(1 / k))
            self.time_sig *= peak_value

    def make_time_array(self, **kwargs):
        if self.sig_type == "Imported":
            self.T = self.time_sig.shape[0] / self.FS
            self.t = np.arange(self.time_sig.shape[0]) / self.FS
        else:
            setattr(self, "T", kwargs.get("T", 5))
            setattr(self, "FS", kwargs.get("FS", 48000))
            # there are default values here. Careful.
            self.t = np.arange(self.T * self.FS) / self.FS

    def normalize(self, **kwargs):
        self.time_sig = self.time_sig / ac.signal.rms(self.time_sig) * kwargs.get("set_RMS", 1)

    def apply_filters(self, **kwargs):
        """
        Need to pass whole filter widget objects to this method.
        Better only pass a dictionary.
        """
        for filter in kwargs.get("filters"):
            filt_type = filter["type"].currentText()
            frequency = filter["frequency"].value()
            order = filter["order"].currentData()
            if filt_type == "HP":
                self.time_sig = ac.signal.highpass(self.time_sig, frequency,
                                                   self.FS, order, zero_phase=False)
            elif filt_type == "LP":
                self.time_sig = ac.signal.lowpass(self.time_sig, frequency,
                                                  self.FS, order, zero_phase=False)
            elif filt_type == "HP (zero phase)":
                self.time_sig = ac.signal.highpass(self.time_sig, frequency,
                                                   self.FS, order//2, zero_phase=True)  # workaround for bug
            elif filt_type == "LP (zero phase)":
                self.time_sig = ac.signal.lowpass(self.time_sig, frequency,
                                                  self.FS, order//2, zero_phase=True)  # workaround for bug
            elif filt_type == "Disabled":
                pass
            else:
                raise KeyError("Unable to apply filter\n", f"Filter type {filt_type} not recognized.")

    def apply_fade_in_out(self):
        n_fade_window = int(min(self.FS / 10, self.T * self.FS / 4))

        # Fade in
        self.time_sig[:n_fade_window] =\
            self.time_sig[:n_fade_window] * make_fade_window_n(0, 1, n_fade_window)

        # Fade out
        self.time_sig[len(self.time_sig) - n_fade_window:] =\
            self.time_sig[len(self.time_sig) - n_fade_window:] * make_fade_window_n(1, 0, n_fade_window)

        self.applied_fade_in_duration = f"{n_fade_window / self.FS * 1000:.0f}ms"


def make_fade_window_n(level_start, level_end, N_total, fade_start_end_idx=None):
    """
    Make a fade-in or fade-out window using information on sample amounts and not time.
    f_start_end defines between which start and stop indexes the fade happens.

    """
    if not fade_start_end_idx:
        fade_start_idx, fade_end_idx = 0, N_total
    else:
        fade_start_idx, fade_end_idx = fade_start_end_idx
    N_fade = fade_end_idx - fade_start_idx

    if N_fade < 0:
        raise ValueError("Fade slice is reverse :(")

    if N_total > 1:
        k = 1 / (N_fade - 1)
        fade_window = (level_start**2 + k * np.arange(N_fade) * (level_end**2 - level_start**2))**0.5
        total_window = np.empty(N_total)

        if fade_start_idx > 0:
            # there are some frames in our output that come before the fade starts
            total_window[:fade_start_idx].fill(level_start)

        if fade_end_idx < N_total:
            # there are some frames in our output that come after the fade ends
            if fade_end_idx > 0:
                total_window[fade_end_idx:].fill(level_end)
            else:
                total_window.fill(level_end)

        if fade_start_idx < N_total and fade_end_idx > 0:
            # some part of the fade window is falling into our [0:N_total] range
            if fade_start_idx >= 0:
                total_window[fade_start_idx:fade_end_idx] = fade_window[:N_total-fade_start_idx]
            elif N_total > fade_end_idx:
                # fade starts before our output starts and ends within our output
                total_window[:fade_end_idx] = fade_window[(0 - fade_start_idx):(fade_end_idx-fade_start_idx)]
            else:
                # fade starts before our output starts and extends further then the end of our output
                total_window[:] = fade_window[(0 - fade_start_idx):(N_total-fade_start_idx)]

    elif N_total <= 1:
        total_window = np.zeros(N_total)

    else:
        raise TypeError("Unknown fade type.")

    return total_window


def make_fade_window_t(level_start, level_end, N_total, FS, fade_start_end_time=None):
    """
    Make a fade-in or fade-out window using time information.
    f_start_end defines between which start and stop times the fade happens.
    All time data in seconds and float.

    """
    if not fade_start_end_time:
        fade_start_time, fade_end_time = 0, N_total / FS
    else:
        fade_start_time, fade_end_time = fade_start_end_time

    fade_start_idx = int(round(fade_start_time * FS))
    fade_end_idx = int(round(fade_end_time * FS))
    fade_start_end_idx = fade_start_idx, fade_end_idx

    return make_fade_window_n(level_start, level_end, N_total, fade_start_end_idx=fade_start_end_idx)


def test_make_fade_window_n():
    import matplotlib.pyplot as plt
    params = [[0.5, 1.5, 10, (-10, -5)],
               [0.5, 1.5, 10, (-5, 5)],
               [0.5, 1.5, 10, (-5, 10)],
               [0.5, 1.5, 10, (4, 6)],
               [0.5, 1.5, 10, (-10, 20)],
               [0.5, 1.5, 10, (5, 15)],
               [0.5, 1.5, 10, (15, 25)],
               [2.5, 1.5, 10, (-10, -5)],
               [2.5, 1.5, 10, (-5, 5)],
               [2.5, 1.5, 10, (4, 6)],
               [2.5, 1.5, 10, (-10, 20)],
               [2.5, 1.5, 10, (5, 15)],
               [2.5, 1.5, 10, (15, 25)],
                [2, 1, 10, (5, 25)],
                [2, 1, 10, (-5, 15)],
                [2, 1, 10, (-15, 5)],
                [1, 0, 10, (0, 10)],

                ]

    for i, param in enumerate(params):
        print(f"Calculating n for param {param}")
        a = make_fade_window_n(*param)
        plt.plot(a**2)
        plt.title(f"Test n {i + 1}: {param}")
        plt.grid()
        plt.show()


def test_make_fade_window_t():
    import matplotlib.pyplot as plt
    params = [[0.5, 1.5, 100, 10, (-10, -5)],
              [0.5, 1.5, 100, 10, (-5, 5)],
              [0.5, 1.5, 100, 10, (-5, 20)],
              [0.5, 1.5, 100, 10, (4, 6)],
              [0.5, 1.5, 100, 10, (-10, 20)],
              [0.5, 1.5, 100, 10, (5, 15)],
              [0.5, 1.5, 100, 10, (15, 25)],
              [2.5, 1.5, 100, 10, (-10, -5)],
              [2.5, 1.5, 100, 10, (-5, 5)],
              [2.5, 1.5, 100, 10, (4, 6)],
              [2.5, 1.5, 100, 10, (-10, 20)],
              [2.5, 1.5, 100, 10, (5, 15)],
              [2.5, 1.5, 100, 10, (15, 25)],
              ]

    for i, param in enumerate(params):
        print(f"Calculating t for param {param}")
        a = make_fade_window_t(*param)
        plt.plot(a**2)
        plt.title(f"Test t {i + 1}: {param}")
        plt.show()


if __name__ == "__main__":
    test_make_fade_window_n()
    # test_make_fade_window_t()
