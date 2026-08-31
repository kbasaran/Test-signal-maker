# Cloud review — branch `0.4.0`

- **Reviewed:** 2026-08-30, `0.4.0` → `main`, 13 files, 647 insertions / 444 deletions
- **Branch tip at review time:** `ccfce94`
- **Reported:** 10 findings — **6 fixed, 3 open, 1 rejected**

Every finding was verified against the code before being acted on. Ordering is by
impact on **audio generation and playback**, not by the reviewer's severity
labels, which did not match: two items it filed as trivial nits sat inside the
signal-synthesis and gain-calibration paths, and one turned out to be an active
bug rather than dead code.

---

# Status at a glance

| # | Finding | Location | Audio impact | Status |
|---|---------|----------|--------------|--------|
| 1 | No stream close at exit | `main.py` | stream left running | **fixed** — `d8b3cb8` |
| 2 | `stop_after` overflow | `main.py` | playback refuses to start | **fixed** — `b044360` |
| 3 | `except RuntimeWarning` + overflow | `main.py` | live: `nan` buffers | **fixed** — `6bf7e67` |
| 4 | `Settings` positional argument | `main.py` | dormant: gain calibration | **fixed** — `6b57666` |
| 5 | Unguarded `player.stream.active` (+3 more) | `main.py` | blocks import workflow | **fixed** — `bf0bec5` |
| 6 | FLAC extension check | `main.py` | filename only | **fixed** — `e7091fe` |
| 7 | `Generator.__init__` dead emit | `main.py:266` | none (UI text) | **open** |
| 8 | Dead "Analyzing…" message | `main.py:1386` | none (UI text) | **open** |
| 9 | `rc<date>` version suffix | `app_config.py:16` | none (labelling) | **open** |
| — | Sample rate not selected on import | `main.py` | — | **rejected, not a bug** |

Findings 4 and 9 are the only two originating on this branch (`4289677`). The
rest predate it. Nothing in the working-tree changes made alongside the review
was flagged.

---

# Open

## 7. `Generator.__init__` emits before any receiver exists — `main.py:264-266`

```python
def __init__(self):
    super().__init__()
    self.signal_not_ready.emit("No signal has been generated.")
```

Qt discards a signal with no connected receivers; there is no replay queue a
later `connect` can drain. The ordering guarantees there are none:

1. `setup_generator_thread()` runs early in `MainWindow.__init__` and constructs
   `Generator()`, which emits immediately.
2. `signal_info_widget` is not created until `main.py:1849`.
3. Both `update_signal_info_widget.connect(...)` (`main.py:1430`) and
   `generator.signal_not_ready.connect(...)` happen in
   `make_connections_and_start_threads`, at the very end of `__init__`.

Verified at runtime: receivers at emit time = 0, and
`signal_info_widget.toPlainText()` is `''` after startup. The panel that should
read "No signal has been generated." is blank until the user does something.

Same pattern as the `Player.__init__` emit dealt with earlier, which is why
`initiate_stream` returns its error message as well as emitting it.
`to_do.txt` already lists "do not have slots inside _init_".

**Plan.** Delete the emit — and with it the whole `__init__`, which then only
calls `super().__init__()` — and give the widget its initial text where it is
created, at `main.py:1849`:

```python
self.signal_info_widget = qtw.QTextEdit("No signal has been generated.",
                                        readOnly=True, parent=self)
```

Setting it at construction means the startup state no longer depends on
connection ordering, which is what broke it. Deliberately *not* routed through
`handler_generator_signal_not_ready`: that handler also clears the plot, calls
`player.stop_play()` and sets `generated_signal = None`, all already the startup
state, and calling it during construction would reintroduce the same ordering
dependency.

## 8. Dead "Analyzing…" message — `main.py:1386`

`update_signal_info_widget.emit("Signal generated. Analyzing...")` is overwritten
synchronously by the final emit a few lines later, before control returns to the
event loop, so it never repaints. The wording is also inaccurate — the spectrum
analysis already completed in the Generator thread before `signal_ready` fired.

**Plan.** Delete the line. The final emit already communicates the finished
state. Not yet re-verified against the current file.

## 9. Every build is labelled `rc<date>` — `config/app_config.py:16`

The comment reads `# uncomment for release candidate builds`, but the line is
live, so `APP_DEFINITIONS["version"]` is always mutated. Verified:
`0.4.0rc260830`.

Flows into the window title, the About dialog, the Windows shortcut label and the
MSI version field (`package_win/setup.py`). Does **not** leak into settings
storage — `Settings.__post_init__` buckets by `"0.4"`.

**Plan — needs a decision, not just a fix.** Three options: comment the line out
so it matches the comment; reword the comment to describe the always-RC
behaviour; or gate it on an env var so no source edit is needed at release time.

---

# Fixed

## 1. No stream close at exit — `d8b3cb8`

`MainWindow.shutdown_audio` aborts and closes the output stream, connected to
`aboutToQuit` **ahead of** the thread quits.

The reviewer's proposed fix — reconnecting `player.stop_play` — is a **no-op**,
measured:

| `aboutToQuit` handler | state after `exec()` | quit cost |
|---|---|---|
| none | `closed=False active=True` | 259 ms |
| `player.stop_play` (reviewer's fix) | `closed=False active=True` | 287 ms |
| abort + close | `closed=True active=False` | 411 ms |

`stop_play` only sets `fade_out_frames` and returns without closing, and it is a
queued call into `player_thread`, whose event loop is already quitting because
`player_thread.quit()` was connected first.

`sounddevice`'s own atexit net does **not** cover this app: `_exit_handler` only
stops and closes `_last_callback.stream`, and `_last_callback` is set exclusively
by `_CallbackContext.start_stream`, i.e. by `sd.play()` / `sd.rec()` /
`sd.playrec()`. This app builds `sd.OutputStream` directly, so it stays `None`
(verified) and the stream was torn down only by the later `Pa_Terminate()`.

**Attribution corrected.** The reviewer claims `eb0377f` ("improve exit
behaviour") *added* the connection and that this PR dropped it. `eb0377f` is the
commit that **removed** it, in the same hunk that created
`make_connections_and_start_threads`. Pre-existing, ~18 months old.

## 2. `stop_after` overflow — `b044360`

Widget `Maximum` lowered from `1e6-1` to `35700` minutes; the tooltip now states
the limit and that it comes from a 32-bit timer interval. The hard boundary is
35791.3 minutes (`INT_MAX / 60000`), confirmed:

```
35000   min -> 2100000000 ms  ok
35791.3 min -> 2147478000 ms  ok
35791.4 min -> 2147484000 ms  OverflowError
```

**Attribution corrected.** Not "newly introduced by this PR" — `eda98d9` is
already contained in `main`.

## 3. `except RuntimeWarning` + sweep overflow — `6bf7e67`

**This was an active bug, not dead code.** The `except` cannot fire (NumPy warns
via `warnings.warn`; no `seterr`/`errstate`/`simplefilter` anywhere), but the
failure it was written for is real: `n = (omega_end/omega_start)**(1/T)`
overflows to `inf`, because `1/T` is `samplerate/frames`.

| block size | `1/T` | overflows when ratio exceeds |
|---|---|---|
| 1024 | 46.9 | 3,767,038 |
| 512 | 93.8 | 1,941 |
| 256 | 187.5 | **44** |
| 128 | 375.0 | **6.6** |
| 64 | 750.0 | **2.6** |

`inf` → `k = 0.0` → `0.0 * (inf-1) = nan` → whole buffer NaN. Measured: a
100 Hz → 1 kHz move on a 128-frame block produced 128/128 NaN.

Fixed by computing `log_n = np.log(omega_end / omega_start) / T` directly — the
large intermediate is never formed — plus an explicit precondition returning the
`np.zeros` fallback the author intended. Bit-identical output where the old code
already worked (max abs diff `0.000e+00`); every failing case now 0 NaN.

⚠️ Commit `33eeaba` closed **GitHub issue #1** with only the guard that never
fired, so that issue was never actually resolved. It may deserve reopening or a
deliberate re-close.

## 4. `Settings` positional argument — `6b57666`

Now `Settings()`. The first dataclass field is `system_gains`, so the positional
string landed there — verified, `system_gains` was literally
`'Test Signal Maker'` during `__init__` before `read_all_from_registry`
overwrote it.

Note the reviewer's alternative `Settings(app_name=...)` is **not** equivalent:
the keyword is discarded as an attribute by `read_all_from_registry` but still
selects the QSettings storage title, so it would silently change which settings
bucket is used.

## 5. Unguarded `player.stream.active` — `bf0bec5`

The reported `None` guard was **one of four defects** in the same six-line loop:

- `self.player.stream.active` on a possibly-`None` stream
- `timer == 99` unreachable inside `range(10)`
- the `else` broke out on the first iteration, so it never polled — measured a
  flat 100–200 ms and one iteration for every input
- an unconditional second `msleep(100)` even when nothing was playing

The wait now lives in `Player.stop_play(blocking=True, timeout_ms=1000)`, merged
with the former `stop_play_blocking` and polling at 10 ms instead of spinning.
Returns `False` on timeout; the three other callers no longer wait forever.

⚠️ `stop_play` is connected to `QPushButton.clicked`, which passes its `bool`
into the first positional parameter — **no form of the `Slot` decorator prevents
this**, verified for `@Slot()`, `@Slot(str)` and `@Slot(bool)`. The connections
therefore drop the argument explicitly:

```python
stop_button.clicked.connect(lambda *_: self.player.stop_play())
```

Without that, making either Stop button checkable would silently turn blocking on
and freeze the GUI.

## 6. FLAC extension check — `e7091fe`

Now uses `Path.suffix`, normalizes the written suffix to lowercase, and appends
rather than replaces so `recording.v2` keeps its ending. Verified by writing real
files and reading them back:

| typed | format | written | reads as |
|---|---|---|---|
| `myfile.FLAC` | FLAC | `myfile.flac` | FLAC |
| `analytics.lac` | FLAC | `analytics.lac.flac` | FLAC |
| `Cadillac` | FLAC | `Cadillac.flac` | FLAC |
| `myfile.WAV` | WAV | `myfile.wav` | WAV |

Rows 2 and 3 previously got no `.flac` at all, because `'FLAC'[-3:]` is `'lac'`.
OGG was deliberately not added to the format combo, though `file_filters` still
defines an unreachable entry for it.

---

# Rejected

## Imported file's sample rate is never selected — `main.py:1342-1347`

Reported as `pre_existing`, and initially ranked first here on the grounds that
it produces audio at the wrong sample rate. **That was wrong — the current
behaviour is intentional and correct.**

The `sample_rate_selector` list `[22050, 44100, 48000, 96000]` is the set of rates
a sound card can be expected to play, not a mirror of the source file. When an
imported file's rate is not in that list, leaving the selection alone is
deliberate: `reuse_existing` → `apply_processing` → `apply_resampling`
(`generictools/signal_tools.py:338-345`) calls `scipy.signal.resample` and updates
`self.FS`. There is no pitch or duration error.

Auto-selecting the file's own rate would be actively harmful: `ugs_play` forces
the output stream to `signal_object.FS`, so the device would be asked to open at
a rate the list exists to avoid.

A fix was applied and then reverted.

Two adjacent things worth a look, independent of this finding:

- `apply_resampling` raises `NotImplementedError("Downsampling of signal not
  implemented.")` when the target rate is below the source. Importing a 48 kHz
  file while 22050 is selected fails at Generate time.
- `addItem` still inserts the unsupported rate into the dropdown, so a user can
  select it manually and hit the device-rejection path.

---

# Related work done alongside, not from the review

- **Thread teardown** (`8a3f471`) — `player_thread` and `generator_thread` were
  `quit()`ed but never `wait()`ed. Quitting while the generator is busy destroys
  a running `QThread`, which Qt turns into `SIGABRT`; reproduced at exit code 134
  for 30 s, 60 s and 120 s signals. `MainWindow.wait_for_threads(timeout_ms=20_000)`
  now waits after `app.exec()`. The 20 s default covers the slowest generation the
  GUI allows (600 s at 96 kHz measures ~14.4 s). **A thread still running when the
  timeout expires still aborts** — the timeout bounds the hang, it does not remove
  the risk. A cancellation flag in `Generator` would.
- **`ugs_play` sample rate** — the stream was rebuilt at `settings.play_sample_rate`
  precisely when it already matched the signal, so the rate alternated between
  presses: worked once, failed on the second. Now always forced to
  `signal_object.FS`, with an early return when the stream cannot be opened.
- **`sweep_play`** — matching single-report error handling.
- **Graph widget** — swapped to `generictools.graphing_widget.MatplotlibWidget`
  with a fixed y-limits policy and an x-limits policy tracking Nyquist.

# Reviewer errors, for the record

1. **Finding 1** — claimed `eb0377f` *added* the `aboutToQuit → stop_play`
   connection. It removed it.
2. **Finding 2** — claimed the `stop_after` maximum was "newly introduced by this
   PR". `eda98d9` is already on `main`.
3. **Finding 3** — filed as a `nit` about dead code; it was concealing an active
   NaN bug.
4. **Finding 5** — reported one defect in a loop that had four.
5. **Rejected finding** — the premise (wrong-pitch playback) was false.

All describe real code states; the blame and the severity were what was off.
