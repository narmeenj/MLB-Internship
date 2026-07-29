# Day-14: Image Processing Toolkit (OpenCV + Gradio)

A menu-driven Python application for common image-processing operations,
built with OpenCV, plus a Gradio web app that puts the same toolkit in a
user-friendly browser UI.

`image_toolkit/toolkit.py` is a refactor of an original standalone batch
script (by Narmeen Javed, operating on `three_kittens.jpg`) into a
reusable, importable module - the hardcoded Windows paths and single-image
assumption were removed, functions were made to raise clear errors instead
of just printing and returning the original image, and a few extras
(`save_image`, `rotate_any`, brightness/contrast, BGR↔RGB, side-by-side)
were added on top so the same functions can power both the CLI menu and
the Gradio app. The original script's exact pipeline is still available
as CLI menu option 14 ("Run full original batch pipeline").

## Folder Structure

```
Day-14/
├── opencv_practice/         # 11 small standalone OpenCV practice scripts
├── image_toolkit/
│   ├── toolkit.py           # Core reusable image-processing functions
│   └── main.py              # Menu-driven CLI application
├── gradio_app/
│   └── app.py                # Gradio web app
├── sample_images/            # 5 sample images used for the challenge task
│   └── generate_samples.py   # Script that generated them
├── processed_outputs/        # Outputs of the challenge task, one folder per image
│   ├── landscape/
│   ├── person/
│   ├── vehicle/
│   ├── document/
│   └── object/
├── run_challenge_task.py     # Applies every operation to all 5 sample images
├── requirements.txt
└── README.md
```

## Credit

Core image-processing logic originally written by **Narmeen Javed** as a
standalone script; refactored here into a reusable module shared by the
CLI and Gradio apps.

## Setup

```bash
python -m venv venv
source venv/bin/activate        # venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## Running the CLI Toolkit

```bash
cd image_toolkit
python main.py
```

You'll get a numbered menu:

1. Load an image (prints height/width/channels/file size, like the
   original script's analysis)
2. Convert to grayscale
3. Resize (exact width/height)
4. Resize to 4 preset sizes at once (small/medium/large/wide)
5. Rotate - fixed 90 / 180 / 270 (fast, lossless `cv2.rotate`)
6. Rotate - any angle (canvas-expanding `cv2.warpAffine`)
7. Flip (horizontal / vertical / both)
8. Crop - custom coordinates
9. Crop into 9 named regions at once (upper/lower/left/right/center/corners)
10. Draw demo shapes (rectangle, circle, line, triangle)
11. Add custom text (defaults to "name - today's date" if left blank)
12. Adjust brightness / contrast
13. Show original vs processed side by side
14. Run the full original batch pipeline on a fresh image
15. Save the current working image
16. Reset to the original loaded image

Operations chain on top of a "working image" kept in memory until you
save or reset.

## Running the Gradio App

```bash
cd gradio_app
python app.py
```

Then open the printed local URL (usually `http://127.0.0.1:7860`). To
share it publicly:

```bash
ngrok http 7860
```

and share the `https://....ngrok-free.app` link.

The Gradio app lets you:
- Upload your own image, or click one of 5 bundled sample images.
- Pick an operation from a dropdown (extra parameter controls appear
  automatically for that operation).
- See the original and processed image side by side.
- Download the processed result.
- Get friendly error messages instead of crashes for any bad input
  (no image uploaded, invalid file, missing text, out-of-range crop, etc.)

## Running the Challenge Task

```bash
python run_challenge_task.py
```

This applies every operation (grayscale, resize, rotate, flip x2, crop,
draw shapes, add text, brightness up, contrast up, side-by-side compare)
to all 5 sample images and writes the results into
`processed_outputs/<category>/`.

---

## BGR vs RGB

OpenCV stores color images with channels in **B**lue-**G**reen-**R**ed
order, while most other libraries and the "natural" way people think
about color (and formats like RGB, PNG viewers, matplotlib, PIL) use
**R**ed-**G**reen-**B**lue order. The pixel values are the same three
numbers - only their *order in the array* differs. If you display an
OpenCV (BGR) image using a viewer that expects RGB, or vice versa,
red and blue appear swapped (skies look orange, skin looks blue, etc.).
That's why any time an image crosses between OpenCV and something like
Gradio, PIL, or matplotlib, you need `cv2.cvtColor(img, cv2.COLOR_BGR2RGB)`
(or the reverse) to keep colors correct. This project's toolkit includes
a dedicated "BGR vs RGB Comparison" operation that shows both versions
side by side so the difference is visible.

## What Are Grayscale Images, and Why Use Them?

A grayscale image stores only **one** channel per pixel - a single
intensity/brightness value (commonly 0-255) - instead of three (B, G, R).
Each pixel represents shades of gray rather than color. Grayscale is
used because:

- It **reduces data size and computation** (1 channel instead of 3),
  which speeds up many algorithms.
- Many classic computer-vision techniques (edge detection, thresholding,
  contour finding, feature matching) operate on intensity information
  and don't need color at all.
- It removes color as a variable when only shape/structure/lighting
  matters, making some analyses more robust.

## OpenCV Functions Used

| Function | Purpose |
|---|---|
| `cv2.imread` | Load an image from disk |
| `cv2.imwrite` | Save an image to disk |
| `cv2.cvtColor` | Color-space conversion (BGR↔Gray, BGR↔RGB) |
| `cv2.resize` | Resize an image (exact size or by scale) |
| `cv2.rotate` | Fast, lossless 90°/180°/270° rotation |
| `cv2.getRotationMatrix2D` / `cv2.warpAffine` | Rotate an image by any angle, with canvas expansion so nothing is clipped |
| `cv2.flip` | Flip an image horizontally / vertically / both |
| NumPy slicing (`img[y1:y2, x1:x2]`) | Crop a region of interest, including 9 preset named regions |
| `cv2.line`, `cv2.rectangle`, `cv2.circle`, `cv2.polylines` | Draw demo shapes (line, rectangle, circle, triangle) |
| `cv2.putText` | Draw custom text onto an image |
| `cv2.convertScaleAbs` | Adjust brightness/contrast via `alpha*img + beta` |
| `np.hstack` | Combine two images side by side for comparison |

## Challenges Faced and How They Were Solved

- **Rotation clipping corners.** A plain `cv2.warpAffine` with the
  original canvas size crops the corners of a rotated image. Solved by
  computing the new bounding-box size from the rotation angle and
  adjusting the translation part of the rotation matrix so the whole
  rotated image fits.
- **Comparing images of different types/sizes side by side.** Grayscale
  results are single-channel and can be a different size after a
  resize/crop, so `np.hstack` alone would fail. Solved by converting any
  grayscale image back to 3 channels and resizing the second image to
  match the first image's height before stacking.
- **Gradio giving RGB but OpenCV expecting BGR.** Every image coming out
  of `gr.Image(type="numpy")` is RGB, but all the toolkit functions
  assume OpenCV's BGR convention. Solved by converting at the boundary:
  RGB→BGR right after the user's upload, and BGR→RGB again right before
  sending anything back to Gradio for display.
- **Making the Gradio app fail gracefully.** Users might submit without
  an image, upload something that isn't a valid image, or leave a
  required field (like the crop size or text) empty/invalid. Solved by
  validating inputs explicitly and wrapping the whole processing
  function in try/except blocks that turn any exception into a short,
  friendly status message instead of letting a traceback reach the UI.
- **No graphical display in a headless/server environment.** The CLI's
  preview windows (`cv2.imshow`) don't just raise a catchable `cv2.error`
  when there's no display - on Linux, a missing Qt/X11 backend can abort
  the whole Python process instead. Wrapping the call in try/except
  wasn't enough. Solved by checking for a real display (`DISPLAY` env var
  on Linux, assumed present on Windows/macOS) *before* ever calling
  `cv2.imshow`, so headless runs just print a note and skip the preview
  while still completing and letting the user save the result.
- **Original script assumed one hardcoded image and folder.** The base
  script pointed at a fixed `E:\...\three_kittens.jpg` path and a fixed
  output folder, so it could only ever process one specific image.
  Solved by turning every operation into a parameterized function in
  `toolkit.py` that takes the image/paths as arguments, so the exact
  same logic now works from the CLI menu, the Gradio app, and the
  batch-challenge script on any image.
- **`rotate_img`'s fixed 90/180/270 rotation isn't enough for a Gradio
  angle slider.** Keeping the original fast `cv2.rotate`-based function
  as-is would mean the UI could only offer 3 rotation choices. Solved by
  adding a separate `rotate_any()` using `cv2.getRotationMatrix2D` /
  `cv2.warpAffine` with an expanded canvas, used wherever a free angle is
  needed, while keeping `rotate_img()` for the fast lossless case.

## Sample Images

Five programmatically generated placeholder images are included in
`sample_images/` (landscape, person, vehicle, document, object) so the
project is fully self-contained. Feel free to swap in your own real
photos with the same file names, or point `run_challenge_task.py` and
`gradio_app/app.py` at different files.

