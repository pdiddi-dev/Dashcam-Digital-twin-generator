# Demo shot list

Turn this into a 45-60 second screen recording (macOS ⌘⇧5 → *Record Selected Portion*). Ships as `docs/demo.gif` and embeds in the top of the README.

## Shots

1. **[0:00–0:05]** Terminal in the repo root. Prompt visible. Type slowly:
   ```
   python build_twin.py input.MOV
   ```
   Enter.

2. **[0:05–0:35]** Pipeline runs. Two log lines appear:
   ```
   [1/2] Extracting scene from input.MOV via NVIDIA VLM...
         -> twins/input/scene.json (7 actors)
   [2/2] Building USD stage...
         -> twins/input/scene.usda
   ```
   Then:
   ```
   Done. Open in a viewer: open twins/input/scene.usda
   ```

3. **[0:35–0:42]** Type:
   ```
   cat twins/input/scene.json | head -30
   ```
   Enter. Structured JSON scrolls by.

4. **[0:42–0:60]** Cmd-Tab to Finder positioned on `twins/input/`. Single-click `scene.usda`. Hit Space for QuickLook. Orbit the 3D scene with the mouse for 3-5 seconds.

## Convert to GIF

If the recording lands as a `.mov`, convert with `ffmpeg` (install once via `brew install ffmpeg`):

```bash
ffmpeg -i ~/Desktop/demo.mov -vf "fps=12,scale=960:-1:flags=lanczos" -loop 0 docs/demo.gif
```

Target: <5 MB gif. If bigger, drop the fps to 8 or the width to 720.

## Add to README

Once `docs/demo.gif` exists, add this line right after the hero image in `README.md`:

```markdown
![Pipeline demo](docs/demo.gif)
```
