import json
from pathlib import Path
import tkinter as tk


DATASET_DIR = Path(__file__).resolve().parents[1]
JSON_DIR = DATASET_DIR / "jsons"


class HandwritingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Handwriting Editor (polling raw)")

        # Кнопки
        btn_frame = tk.Frame(root)
        btn_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        self.save_btn = tk.Button(btn_frame, text="Save", command=self.save_data)
        self.save_btn.pack(side=tk.LEFT, padx=2)

        self.undo_btn = tk.Button(btn_frame, text="Undo", command=self.undo_last_stroke)
        self.undo_btn.pack(side=tk.LEFT, padx=2)

        self.clear_btn = tk.Button(btn_frame, text="Clear", command=self.clear_all)
        self.clear_btn.pack(side=tk.LEFT, padx=2)

        # Холст
        self.canvas = tk.Canvas(root, bg="black", cursor="cross")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Configure>", self.redraw_grid)

        # Данные
        self.strokes = []               # готовые штрихи
        self.current_stroke_points = [] # [[x,y,state], ...]
        self.current_visual_points = [] # для линии (x,y)
        self.current_line_id = None
        self.drawing = False
        self.last_x = None              # последние опрошенные координаты
        self.last_y = None
        self.poll_id = None             # идентификатор таймера опроса

        # Мышь / перо – только начало и конец, а движение опрашиваем сами
        self.canvas.bind("<Button-1>", self.on_press)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)

    # ---------- сетка ----------
    def redraw_grid(self, event=None):
        self.canvas.delete("grid")
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        for x in range(0, w, 50):
            self.canvas.create_line(x, 0, x, h, fill="gray", dash=(2, 4), tags="grid")
        for y in range(0, h, 50):
            self.canvas.create_line(0, y, w, y, fill="gray", dash=(2, 4), tags="grid")

    # ---------- опрос позиции ----------
    def poll_motion(self):
        """Периодически проверяет позицию курсора, пока drawing=True."""
        if not self.drawing:
            self.poll_id = None
            return
        # Получаем глобальные экранные координаты и преобразуем в координаты холста
        abs_x = self.canvas.winfo_pointerx()
        abs_y = self.canvas.winfo_pointery()
        # Переводим относительно холста
        x = abs_x - self.canvas.winfo_rootx()
        y = abs_y - self.canvas.winfo_rooty()

        # Если положение изменилось – добавляем точку
        if x != self.last_x or y != self.last_y:
            self.last_x = x
            self.last_y = y
            self.current_stroke_points.append([x, y, 0])
            self.current_visual_points.append((x, y))

            # Обновляем визуальную линию
            flat = [coord for p in self.current_visual_points for coord in p]
            self.canvas.coords(self.current_line_id, *flat)

        # Продолжаем опрос через 10 мс
        self.poll_id = self.root.after(10, self.poll_motion)

    # ---------- рисование ----------
    def on_press(self, event):
        self.drawing = True
        x, y = event.x, event.y
        self.last_x = x
        self.last_y = y

        self.current_stroke_points = [[x, y, 0]]
        self.current_visual_points = [(x, y)]

        # Временная линия
        self.current_line_id = self.canvas.create_line(
            x, y, x, y, fill="red", width=2, tags="temp"
        )

        # Запускаем опрос позиции
        self.poll_id = self.root.after(10, self.poll_motion)

    def on_release(self, event):
        if not self.drawing:
            return
        self.drawing = False

        # Останавливаем опрос
        if self.poll_id:
            self.root.after_cancel(self.poll_id)
            self.poll_id = None

        x, y = event.x, event.y

        # Добавляем завершающую точку pen-up (state=1)
        self.current_stroke_points.append([x, y, 1])
        self.canvas.itemconfig(self.current_line_id, tags=("stroke",))

        # Если за всё время была только точка нажатия (ни одной промежуточной)
        if len(self.current_stroke_points) == 2:
            # Одиночный клик → рисуем точку вместо линии (как в оригинале Kivy)
            self.canvas.delete(self.current_line_id)
            r = 3
            oval_id = self.canvas.create_oval(
                x - r, y - r, x + r, y + r, fill="red", outline=""
            )
            stroke_data = {
                "points": self.current_stroke_points,
                "canvas_ids": [oval_id]
            }
        else:
            stroke_data = {
                "points": self.current_stroke_points,
                "canvas_ids": [self.current_line_id]
            }

        self.strokes.append(stroke_data)

        # Сброс
        self.current_stroke_points = []
        self.current_visual_points = []
        self.current_line_id = None
        self.last_x = None
        self.last_y = None

    # ---------- кнопки ----------
    def save_data(self):
        if not self.strokes:
            print("Нет данных для сохранения")
            return
        all_points = []
        for s in self.strokes:
            all_points.extend(s["points"])

        JSON_DIR.mkdir(parents=True, exist_ok=True)
        idx = 1
        while (JSON_DIR / f"trajectory_{idx}.json").exists():
            idx += 1
        filename = JSON_DIR / f"trajectory_{idx}.json"
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(all_points, f, ensure_ascii=False)
        print(f"Сохранено {len(all_points)} точек в {filename}")
        self.clear_all()

    def undo_last_stroke(self):
        if not self.strokes:
            print("Нечего отменять")
            return
        last = self.strokes.pop()
        for item_id in last["canvas_ids"]:
            self.canvas.delete(item_id)
        print("Последний штрих удалён")

    def clear_all(self):
        for s in self.strokes:
            for item_id in s["canvas_ids"]:
                self.canvas.delete(item_id)
        self.strokes = []
        if self.current_line_id:
            self.canvas.delete(self.current_line_id)
            self.current_line_id = None
        self.drawing = False
        print("Холст очищен")


if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("800x600")
    app = HandwritingApp(root)
    root.mainloop()
