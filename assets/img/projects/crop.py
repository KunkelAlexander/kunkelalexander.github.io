import os
import sys
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


class CropGalleryApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Crop Gallery")
        self.root.geometry("1100x800")

        self.folder = None
        self.image_paths = []
        self.index = 0

        self.original_image = None
        self.display_image = None
        self.tk_image = None

        self.scale = 1.0
        self.offset_x = 0
        self.offset_y = 0

        self.crop_rect = None
        self.rect_id = None

        self.drag_mode = None
        self.drag_start = None
        self.rect_start = None

        self.handle_size = 10
        self.min_rect_size = 10

        self.build_ui()
        self.choose_folder()

    def build_ui(self):
        top = tk.Frame(self.root)
        top.pack(side=tk.TOP, fill=tk.X, padx=8, pady=6)

        self.folder_button = tk.Button(top, text="Choose folder", command=self.choose_folder)
        self.folder_button.pack(side=tk.LEFT)

        self.prev_button = tk.Button(top, text="← Previous", command=self.previous_image)
        self.prev_button.pack(side=tk.LEFT, padx=(10, 0))

        self.save_next_button = tk.Button(top, text="Save crop →", command=self.save_crop_and_next)
        self.save_next_button.pack(side=tk.LEFT, padx=(10, 0))

        self.next_button = tk.Button(top, text="Skip →", command=self.next_image)
        self.next_button.pack(side=tk.LEFT, padx=(10, 0))

        self.reset_button = tk.Button(top, text="Reset selection", command=self.reset_selection)
        self.reset_button.pack(side=tk.LEFT, padx=(10, 0))

        self.info_label = tk.Label(top, text="", anchor="w")
        self.info_label.pack(side=tk.LEFT, padx=20)

        self.pixel_label = tk.Label(self.root, text="", anchor="w", font=("TkDefaultFont", 10))
        self.pixel_label.pack(side=tk.TOP, fill=tk.X, padx=8)

        canvas_frame = tk.Frame(self.root)
        canvas_frame.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(canvas_frame, bg="#222222", cursor="crosshair")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.canvas.bind("<Configure>", lambda event: self.reload_current_image())
        self.canvas.bind("<ButtonPress-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)

        self.root.bind("<Right>", lambda event: self.save_crop_and_next())
        self.root.bind("<Left>", lambda event: self.previous_image())
        self.root.bind("<Escape>", lambda event: self.root.quit())

    def choose_folder(self):
        folder = filedialog.askdirectory(title="Choose image folder")
        if not folder:
            if not self.image_paths:
                self.root.destroy()
            return

        self.folder = Path(folder)
        self.image_paths = sorted(
            p for p in self.folder.iterdir()
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
        )

        if not self.image_paths:
            messagebox.showerror("No images found", "No JPEG, PNG, or WEBP images found in this folder.")
            return

        self.index = 0
        self.load_image()

    def load_image(self):
        if not self.image_paths:
            return

        path = self.image_paths[self.index]

        try:
            self.original_image = Image.open(path)
            self.original_image.load()
        except Exception as exc:
            messagebox.showerror("Could not open image", f"{path.name}\n\n{exc}")
            self.next_image()
            return

        self.crop_rect = None
        self.reload_current_image()

    def reload_current_image(self):
        if self.original_image is None:
            return

        canvas_w = max(self.canvas.winfo_width(), 1)
        canvas_h = max(self.canvas.winfo_height(), 1)

        img_w, img_h = self.original_image.size

        margin = 20
        max_w = max(canvas_w - margin * 2, 1)
        max_h = max(canvas_h - margin * 2, 1)

        self.scale = min(max_w / img_w, max_h / img_h, 1.0)

        display_w = max(int(img_w * self.scale), 1)
        display_h = max(int(img_h * self.scale), 1)

        self.display_image = self.original_image.resize(
            (display_w, display_h),
            Image.Resampling.LANCZOS
        )

        self.tk_image = ImageTk.PhotoImage(self.display_image)

        self.offset_x = (canvas_w - display_w) // 2
        self.offset_y = (canvas_h - display_h) // 2

        self.canvas.delete("all")
        self.canvas.create_image(self.offset_x, self.offset_y, anchor=tk.NW, image=self.tk_image)

        if self.crop_rect is None:
            self.crop_rect = [0, 0, img_w, img_h]

        self.draw_crop_rect()
        self.update_labels()

    def draw_crop_rect(self):
        self.canvas.delete("crop")
        if self.crop_rect is None:
            return

        x1, y1, x2, y2 = self.image_to_canvas_rect(self.crop_rect)

        self.rect_id = self.canvas.create_rectangle(
            x1, y1, x2, y2,
            outline="yellow",
            width=2,
            tags="crop"
        )

        # Darken outside the crop area
        img_x1 = self.offset_x
        img_y1 = self.offset_y
        img_x2 = self.offset_x + self.display_image.width
        img_y2 = self.offset_y + self.display_image.height

        overlay_fill = "#000000"
        stipple = "gray50"

        self.canvas.create_rectangle(img_x1, img_y1, img_x2, y1, fill=overlay_fill, stipple=stipple, outline="", tags="crop")
        self.canvas.create_rectangle(img_x1, y2, img_x2, img_y2, fill=overlay_fill, stipple=stipple, outline="", tags="crop")
        self.canvas.create_rectangle(img_x1, y1, x1, y2, fill=overlay_fill, stipple=stipple, outline="", tags="crop")
        self.canvas.create_rectangle(x2, y1, img_x2, y2, fill=overlay_fill, stipple=stipple, outline="", tags="crop")

        self.canvas.tag_raise(self.rect_id)

        for hx, hy in self.get_handle_positions_canvas():
            self.canvas.create_rectangle(
                hx - self.handle_size // 2,
                hy - self.handle_size // 2,
                hx + self.handle_size // 2,
                hy + self.handle_size // 2,
                fill="yellow",
                outline="black",
                tags="crop"
            )

    def get_handle_positions_canvas(self):
        x1, y1, x2, y2 = self.image_to_canvas_rect(self.crop_rect)
        return [
            (x1, y1), ((x1 + x2) // 2, y1), (x2, y1),
            (x1, (y1 + y2) // 2), (x2, (y1 + y2) // 2),
            (x1, y2), ((x1 + x2) // 2, y2), (x2, y2),
        ]

    def image_to_canvas_rect(self, rect):
        x1, y1, x2, y2 = rect
        return [
            int(self.offset_x + x1 * self.scale),
            int(self.offset_y + y1 * self.scale),
            int(self.offset_x + x2 * self.scale),
            int(self.offset_y + y2 * self.scale),
        ]

    def canvas_to_image_point(self, cx, cy):
        ix = int(round((cx - self.offset_x) / self.scale))
        iy = int(round((cy - self.offset_y) / self.scale))

        img_w, img_h = self.original_image.size
        ix = max(0, min(img_w, ix))
        iy = max(0, min(img_h, iy))

        return ix, iy

    def on_mouse_down(self, event):
        if self.original_image is None or self.crop_rect is None:
            return

        self.drag_start = (event.x, event.y)
        self.rect_start = self.crop_rect.copy()

        self.drag_mode = self.detect_drag_mode(event.x, event.y)

        if self.drag_mode is None:
            ix, iy = self.canvas_to_image_point(event.x, event.y)
            self.crop_rect = [ix, iy, ix, iy]
            self.drag_mode = "new"
            self.rect_start = self.crop_rect.copy()

    def on_mouse_drag(self, event):
        if not self.drag_mode or self.crop_rect is None:
            return

        img_w, img_h = self.original_image.size
        ix, iy = self.canvas_to_image_point(event.x, event.y)

        x1, y1, x2, y2 = self.rect_start

        if self.drag_mode == "new":
            sx, sy = self.rect_start[0], self.rect_start[1]
            self.crop_rect = self.normalize_rect([sx, sy, ix, iy])

        elif self.drag_mode == "move":
            start_ix, start_iy = self.canvas_to_image_point(*self.drag_start)
            dx = ix - start_ix
            dy = iy - start_iy

            width = x2 - x1
            height = y2 - y1

            nx1 = max(0, min(img_w - width, x1 + dx))
            ny1 = max(0, min(img_h - height, y1 + dy))

            self.crop_rect = [nx1, ny1, nx1 + width, ny1 + height]

        else:
            nx1, ny1, nx2, ny2 = x1, y1, x2, y2

            if "left" in self.drag_mode:
                nx1 = ix
            if "right" in self.drag_mode:
                nx2 = ix
            if "top" in self.drag_mode:
                ny1 = iy
            if "bottom" in self.drag_mode:
                ny2 = iy

            self.crop_rect = self.normalize_rect([nx1, ny1, nx2, ny2])

        self.enforce_min_size()
        self.draw_crop_rect()
        self.update_labels()

    def on_mouse_up(self, event):
        self.drag_mode = None
        self.drag_start = None
        self.rect_start = None

    def detect_drag_mode(self, cx, cy):
        x1, y1, x2, y2 = self.image_to_canvas_rect(self.crop_rect)

        handles = {
            "top-left": (x1, y1),
            "top": ((x1 + x2) // 2, y1),
            "top-right": (x2, y1),
            "left": (x1, (y1 + y2) // 2),
            "right": (x2, (y1 + y2) // 2),
            "bottom-left": (x1, y2),
            "bottom": ((x1 + x2) // 2, y2),
            "bottom-right": (x2, y2),
        }

        for mode, (hx, hy) in handles.items():
            if abs(cx - hx) <= self.handle_size and abs(cy - hy) <= self.handle_size:
                return mode

        if x1 <= cx <= x2 and y1 <= cy <= y2:
            return "move"

        return None

    def normalize_rect(self, rect):
        x1, y1, x2, y2 = rect
        return [
            min(x1, x2),
            min(y1, y2),
            max(x1, x2),
            max(y1, y2),
        ]

    def enforce_min_size(self):
        x1, y1, x2, y2 = self.crop_rect
        img_w, img_h = self.original_image.size

        if x2 - x1 < self.min_rect_size:
            x2 = min(img_w, x1 + self.min_rect_size)

        if y2 - y1 < self.min_rect_size:
            y2 = min(img_h, y1 + self.min_rect_size)

        self.crop_rect = [x1, y1, x2, y2]

    def reset_selection(self):
        if self.original_image is None:
            return

        img_w, img_h = self.original_image.size
        self.crop_rect = [0, 0, img_w, img_h]
        self.draw_crop_rect()
        self.update_labels()

    def save_crop_and_next(self):
        if self.original_image is None or self.crop_rect is None:
            return

        path = self.image_paths[self.index]
        x1, y1, x2, y2 = map(int, self.crop_rect)

        if x2 <= x1 or y2 <= y1:
            messagebox.showwarning("Invalid crop", "Crop selection is empty.")
            return

        try:
            cropped = self.original_image.crop((x1, y1, x2, y2))

            suffix = path.suffix.lower()

            if suffix in {".jpg", ".jpeg"}:
                if cropped.mode in {"RGBA", "LA", "P"}:
                    cropped = cropped.convert("RGB")
                cropped.save(path, quality=95, optimize=True)

            elif suffix == ".png":
                cropped.save(path, optimize=True)

            elif suffix == ".webp":
                cropped.save(path, quality=95, method=6)

            else:
                cropped.save(path)

        except Exception as exc:
            messagebox.showerror("Save failed", f"Could not overwrite image:\n{path.name}\n\n{exc}")
            return

        self.next_image()

    def previous_image(self):
        if not self.image_paths:
            return

        self.index = max(0, self.index - 1)
        self.load_image()

    def next_image(self):
        if not self.image_paths:
            return

        if self.index >= len(self.image_paths) - 1:
            messagebox.showinfo("Done", "Reached the last image.")
            self.load_image()
            return

        self.index += 1
        self.load_image()

    def update_labels(self):
        if self.original_image is None:
            return

        path = self.image_paths[self.index]
        img_w, img_h = self.original_image.size

        self.info_label.config(
            text=f"{self.index + 1}/{len(self.image_paths)}  |  {path.name}  |  {img_w}×{img_h}px"
        )

        if self.crop_rect:
            x1, y1, x2, y2 = map(int, self.crop_rect)
            crop_w = x2 - x1
            crop_h = y2 - y1

            self.pixel_label.config(
                text=f"Crop selection: x={x1}, y={y1}, width={crop_w}, height={crop_h} "
                     f"| box=({x1}, {y1}, {x2}, {y2})"
            )


def main():
    root = tk.Tk()
    app = CropGalleryApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()