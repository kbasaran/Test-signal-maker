# Cloud review — branch `0.4.0`

- **Date:** 2026-08-30
- **Scope:** `0.4.0` → `main`, 13 files, 647 insertions / 444 deletions
- **Branch tip at review time:** `ccfce94`
- **Findings:** 10 (1 normal, 1 pre-existing, 8 nit as classified by the reviewer)

Every finding below was independently verified against the code before this file
was written. Two of the reviewer's attributions were wrong and are corrected in
place — see **Reviewer errors** at the end.

Ranking here is by actual impact, which differs from the reviewer's severity
labels.

---

## Worth fixing

### 1. Imported file's sample rate is never selected — `main.py:1342-1347`

Reviewer severity: `pre_existing`. **Under-rated — this is the most consequential
finding.**

When the imported file's `FS` is not already in `sample_rate_selector`,
`findData` returns `-1` and the code calls `addItem(...)` but never selects the
new entry. `QComboBox.addItem` only auto-selects when the combobox is empty, and
this one is pre-populated at construction with `[22050, 44100, 48000, 96000]`
(`main.py:1449-1452`).

Verified:

```
before addItem, currentData = 44100
after  addItem, currentData = 44100   (imported file was 32000)
```

**Effect:** importing e.g. a 32 kHz file adds "32000" to the dropdown, leaves the
selection on the previous rate, and the next Generate silently rebuilds the
signal at the wrong `FS`. Wrong audio output, no warning. Bites any file whose
rate is outside the four defaults (32000, 88200, 176400, 192000 …).

**Fix:** select the new entry after adding it.

```python
if self.sample_rate_selector.findData(imported_signal.FS) == -1:
    self.sample_rate_selector.addItem(str(imported_signal.FS), imported_signal.FS)
self.sample_rate_selector.setCurrentIndex(
    self.sample_rate_selector.findData(imported_signal.FS))
```

Pre-existing (already on `main`).

---

### 2. Nothing stops the audio stream at exit — `main.py:1382-1383`

Reviewer severity: `normal`.

Only `player_thread.quit` and `generator_thread.quit` are connected to
`aboutToQuit`. There is no `closeEvent` and no `Player.__del__` anywhere in
`main.py` (verified by grep). `QThread.quit()` asks the Qt event loop to stop but
does not touch the PortAudio callback thread, which keeps dereferencing `Player`
attributes (`self.stream`, `self.play_pos`, `self.user_gen_signal`) while Qt
tears down.

**Effect:** possible crash-on-exit, audio continuing briefly past quit, or
PortAudio holding the device — varies by host API.

**Fix:** re-add alongside the existing `.quit` connections.

```python
qtw.QApplication.instance().aboutToQuit.connect(self.player.stop_play)
```

**Attribution corrected.** The reviewer claims commit `eb0377f` ("improve exit
behaviour") is where this connection was *added*, and that the
`setup_player_thread` / `make_connections_and_start_threads` split in this PR
dropped it. The opposite is true: `eb0377f` (Feb 2025) is the commit that
**removed** it, in the same hunk that created
`make_connections_and_start_threads` and moved the two `.quit` connections
across. It reads as an accidental drop during that move. Pre-existing, ~18
months old, already on `main` — not from this branch.

---

### 3. `stop_after` above ~35,791 minutes raises `OverflowError` — `main.py:1564-1565`

Reviewer severity: `nit`.

`Maximum=1e6-1` (minutes) admits values that overflow the signed 32-bit int
argument of `QTimer.setInterval`. `BasicCountDownTimer.__init__`
(`main.py:1240`) does `setInterval(total_duration * 1000)`.

Verified at the boundary:

```
35000 min -> setInterval(2100000000) ok
35791 min -> setInterval(2147460000) ok
36000 min -> OverflowError
999999 min -> OverflowError
```

**Effect:** for roughly 35,792–999,999 minutes the Play button pops an error
instead of playing. The exception is caught by `ugs_play`'s handler, so no crash.
Implausible input in practice.

**Fix:** cap `Maximum` below ~35,000, or switch the input unit to hours.

**Attribution corrected.** The reviewer calls this "a genuine regression newly
introduced by this PR". It is not — `git branch --contains eda98d9` ("allow for
longer stop_after times in player") lists `main`, so it is already released.

---

### 4. `self.player.stream.active` without a `None` guard — `main.py:1975`

Reviewer severity: `nit`.

The only remaining unguarded `player.stream` access in the file (verified by
grep). `Player.stream` is nullable: `initiate_stream` sets it to `None` on
failure and `Player.__init__` does not raise, so the app starts with
`stream is None` if the device cannot be opened.

**Effect:** with a broken audio configuration, selecting "Imported" raises
`AttributeError: 'NoneType' object has no attribute 'active'`, caught and shown
as a misleading "File import failed." popup — blocking a workflow that needs no
audio at all.

**Fix:**

```python
if self.player.stream is not None and self.player.stream.active:
```

Pre-existing (`e738438`).

---

## Minor

### 5. FLAC extension check misses 4-letter suffix — `main.py:1953`

The check compares the last three characters of the filename against the last
three of the format name. `'FLAC'[-3:]` is `'lac'`, so:

```
analytics.lac   fmt=FLAC   treated_as_having_ext=True    <-- wrong
song            fmt=FLAC   treated_as_having_ext=False
song.flac       fmt=FLAC   treated_as_having_ext=True
a.wav           fmt=WAV    treated_as_having_ext=True
```

A file named `analytics.lac` is saved without a `.flac` extension. Narrow edge
case. Fix by using `Path.suffix` / `with_suffix` instead of slicing.

Pre-existing (`9d59f4a`).

---

### 6. `except RuntimeWarning` never fires — `main.py:605-611`

NumPy emits `RuntimeWarning` through `warnings.warn` and returns `inf`/`nan`; it
does not raise. There is no `np.seterr`, `np.errstate` or
`warnings.simplefilter` anywhere in the repo (verified by grep), so the except
clause is unreachable.

Verified: `1000.0 / np.log(1.0)` returns `inf`, no exception.

Harmless today — the caller gate at `main.py:749` (`target_omega > 0 and
self._omega_last > 0 and target_omega != self._omega_last`) rules out every input
the guard was written for. But commit `e0c4f2e` ("bugfix sweep argument guards")
implies a fix that does not exist. Either wrap in `with np.errstate(all="raise"):`
or replace with explicit input checks.

Pre-existing (`33eeaba`).

---

### 7. `Settings(APP_DEFINITIONS["app_name"])` binds to the wrong field — `main.py:2247`

`Settings` is a dataclass whose first field is `system_gains`, so the positional
string lands there, not in `app_name`. Masked twice over: `app_name`'s own
default already equals `APP_DEFINITIONS["app_name"]`, and `__post_init__` calls
`read_all_from_registry()` which overwrites `system_gains` immediately. No
runtime effect today, but the call reads as the opposite of what it does.

**Fix:** `Settings()` or `Settings(app_name=APP_DEFINITIONS["app_name"])`.

**From this branch** (`4289677`).

---

### 8. Every build is labelled `rc<date>` — `config/app_config.py:16`

The comment above reads `# uncomment for release candidate builds`, but the line
is live, so `APP_DEFINITIONS["version"]` is always mutated. Verified:
`version = 0.4.0rc260830`.

Flows into the window title (`main.py:1414`), the About dialog
(`main.py:1782-1786`), the Windows shortcut label and the MSI version field
(`package_win/setup.py:52,59`). Does **not** leak into settings storage —
`Settings.__post_init__` buckets by `"0.4"`.

**Fix:** comment the line to match the comment, reword the comment to match the
code, or gate on an env var.

**From this branch** (`4289677`).

---

### 9. `Generator.__init__` emits before any receiver exists — `main.py:264-266`

`signal_not_ready.emit("No signal has been generated.")` runs during
construction; the connection is only made later in
`make_connections_and_start_threads` (`main.py:1361`). Qt discards signals with
no connected slots, so `signal_info_widget` starts blank.

Found independently during this session while tracing why `generated_signal` is
never initialised — the same pattern also applies to the `signal_exception` emit
in `Player.initiate_stream` when called from `Player.__init__`, which is why that
method returns its error message as well as emitting it.

`to_do.txt` already lists "do not have slots inside _init_".

Pre-existing (`a28f8a4`).

---

### 10. Dead "Analyzing…" message — `main.py:1319`

`update_signal_info_widget.emit("Signal generated. Analyzing...")` is overwritten
synchronously by the final emit at `main.py:1327` before control returns to the
event loop, so it never repaints. The wording is also inaccurate — the spectrum
analysis already completed in the Generator thread before `signal_ready` fired.

Pre-existing (`a28f8a4`).

---

## Attribution summary

| # | Finding | Origin |
|---|---------|--------|
| 1 | Sample rate not selected on import | pre-existing |
| 2 | No `stop_play` at exit | pre-existing (`eb0377f`, Feb 2025) |
| 3 | `stop_after` overflow | pre-existing (`eda98d9`, on `main`) |
| 4 | Unguarded `player.stream.active` | pre-existing (`e738438`) |
| 5 | FLAC extension check | pre-existing (`9d59f4a`) |
| 6 | Dead `except RuntimeWarning` | pre-existing (`33eeaba`) |
| 7 | `Settings` positional argument | **this branch** (`4289677`) |
| 8 | `rc<date>` version suffix | **this branch** (`4289677`) |
| 9 | `Generator.__init__` dead emit | pre-existing (`a28f8a4`) |
| 10 | Dead "Analyzing…" message | pre-existing (`a28f8a4`) |

Only **7** and **8** originate from commits unique to this branch. Nothing from
the working-tree changes made alongside this review was flagged.

## Reviewer errors

1. **Finding 2** — claimed `eb0377f` *added* the `aboutToQuit → stop_play`
   connection and that this PR dropped it. `eb0377f` is the commit that removed
   it.
2. **Finding 3** — claimed the `stop_after` maximum is "newly introduced by this
   PR". `eda98d9` is already contained in `main`.

Both findings describe real code states; only the blame is wrong.

## Suggested order of work

Fix **1**, **2** and **4** first — real user-visible failures, all small changes.
Then **7** and **8**, since they originate on this branch. The rest are cleanup.
