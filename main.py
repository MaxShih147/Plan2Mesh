import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QVBoxLayout, QHBoxLayout,
    QPushButton, QWidget, QCheckBox, QGroupBox, QFileDialog, QSlider, QListWidget, QListWidgetItem,
    QSizePolicy
)
from PyQt5.QtCore import Qt, QSize, QTimer
from PyQt5.QtGui import QPixmap, QImage


class PolygonSimplifierApp(QMainWindow):
    def __init__(self, image_path):
        super().__init__()
        self.image_path = image_path
        self.original_image = cv2.imread(image_path)
        if self.original_image is None:
            raise FileNotFoundError(f"Cannot load image: {image_path}")
        self.processed_image = self.original_image.copy()

        # 預設參數
        self.threshold_value = 200
        self.show_vertices = False  # 是否顯示頂點
        self.fill_alpha = 100  # 內部填充透明度（固定）

        # 儲存 Checkbox 狀態
        self.contour_checkboxes = {}
        self.checkbox_states = {}  # 儲存 Checkbox 的狀態，格式為 {contour_id: True/False}

        # 初始化 UI
        self.init_ui()
        self.update_processing()

    def init_ui(self):
        self.setWindowTitle("Polygon Detection and Filling")
        self.setGeometry(100, 100, 1400, 800)  # 調整視窗大小
        self.setMinimumSize(1400, 800)

        # 圖像顯示區域
        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setScaledContents(True)

        # 列表欄位：顯示輪廓及內洞關係
        self.contour_list = QListWidget(self)  # 指定父層為主視窗
        self.contour_list.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)  # 開啟水平滾動條
        # self.contour_list.currentRowChanged.connect(self.update_processing)  # 連接選中行信號
        self.contour_list.currentRowChanged.connect(self.on_row_changed)

        # 參數調整區域
        sliders_layout = QVBoxLayout()
        sliders_layout.addWidget(self.create_slider("Threshold", 0, 255, self.threshold_value, self.update_processing))

        # Checkbox：是否顯示頂點
        self.vertex_checkbox = QCheckBox("Show Vertices")
        self.vertex_checkbox.setChecked(self.show_vertices)
        self.vertex_checkbox.stateChanged.connect(self.update_processing)
        sliders_layout.addWidget(self.vertex_checkbox)

        # GroupBox：參數區域
        sliders_group = QGroupBox("Parameters")
        sliders_group.setLayout(sliders_layout)

        # 按鈕區域
        buttons_layout = QVBoxLayout()
        load_button = QPushButton("Load Image", self)
        load_button.clicked.connect(self.load_new_image)
        save_button = QPushButton("Save", self)
        save_button.clicked.connect(self.save_results)
        quit_button = QPushButton("Quit", self)
        quit_button.clicked.connect(self.close)
        buttons_layout.addWidget(load_button)
        buttons_layout.addWidget(save_button)
        buttons_layout.addWidget(quit_button)

        sliders_layout.addLayout(buttons_layout)

        # 右側垂直堆疊：參數區域 + 列表
        right_layout = QVBoxLayout()
        right_layout.addWidget(sliders_group, stretch=1)
        right_layout.addWidget(self.contour_list, stretch=2)  # 列表佔較大空間

        # 主佈局：左側圖片 + 右側垂直堆疊區域
        main_layout = QHBoxLayout()
        main_layout.addWidget(self.image_label, stretch=3)  # 圖片佔 3/5 的空間
        main_layout.addLayout(right_layout, stretch=2)  # 控制區域佔 2/5 的空間

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

    def create_slider(self, label, min_val, max_val, init_val, callback):
        layout = QVBoxLayout()
        slider_label = QLabel(f"{label}: {init_val}")
        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(min_val)
        slider.setMaximum(max_val)
        slider.setValue(init_val)
        slider.valueChanged.connect(lambda value: self.on_slider_change(value, slider_label, label, callback))
        layout.addWidget(slider_label)
        layout.addWidget(slider)
        container = QWidget()
        container.setLayout(layout)
        return container

    def on_slider_change(self, value, label, slider_name, callback):
        label.setText(f"{slider_name}: {value}")
        setattr(self, f"{slider_name.lower().replace(' ', '_')}_value", value)
        callback()

    def on_checkbox_state_changed(self, contour_id, state):
        """
        處理 Checkbox 狀態變化。
        """
        is_checked = state == Qt.Checked  # 判斷是否勾選
        print(f"Checkbox for Contour {contour_id} changed to {'Checked' if is_checked else 'Unchecked'}")
        self.update_processing()  # 更新畫面

    def update_checkbox_state(self, contour_id, state):
        """
        更新狀態表中對應 Checkbox 的狀態。
        """
        is_checked = state == Qt.Checked
        self.checkbox_states[contour_id] = is_checked  # 更新狀態表
        print(f"Checkbox state for Contour {contour_id} updated to {'Checked' if is_checked else 'Unchecked'}")
        self.update_processing()  # 重新繪製畫面

    def force_row_update(self):
        """強制觸發 `currentRowChanged` 事件"""
        current_row = self.contour_list.currentRow()
        if current_row != -1:
            temp_row = 0 if current_row != 0 else 1  # 避免 out of range
            print(f"[DEBUG] Forcing row change: {current_row} -> {temp_row} -> {current_row}")
            self.contour_list.setCurrentRow(temp_row)
            self.contour_list.setCurrentRow(current_row)

    def on_row_changed(self, current_row):
        """當選中行變更時觸發"""
        print(f"[DEBUG] on_row_changed() called with current_row: {current_row}")

        if current_row == -1:
            print("[DEBUG] No row selected.")
            return

        item = self.contour_list.item(current_row)
        if item is None:
            print("[DEBUG] item is None, skipping update.")
            return

        # 嘗試獲取 item 對應的 QWidget
        widget = self.contour_list.itemWidget(item)
        if widget is None:
            print("[DEBUG] itemWidget() is None, skipping update.")
            return

        # 從 QWidget 內部取得 QLabel 的文字
        label = widget.layout().itemAt(1).widget()  # QLabel 在 HBoxLayout 的第二個位置
        if label is None:
            print("[DEBUG] QLabel not found in widget, skipping update.")
            return

        contour_text = label.text().strip()
        print(f"[DEBUG] Selected row text from QLabel: '{contour_text}'")

        if not contour_text:
            print("[DEBUG] Empty row text, skipping update.")
            return

        try:
            if "Contour =" in contour_text:
                contour_id_part = contour_text.split('=')[1].strip()
                contour_id = int(contour_id_part.split(',')[0])
                print(f"[DEBUG] Extracted Contour ID: {contour_id}")
            else:
                print("[DEBUG] Contour text format is incorrect.")
                return
        except Exception as e:
            print(f"[DEBUG] Error parsing contour text: {e}")
            return

        if hasattr(self, "selected_contour_id") and self.selected_contour_id == contour_id:
            print("[DEBUG] Contour ID has not changed, skipping update.")
            return

        self.selected_contour_id = contour_id

        print("[DEBUG] Calling update_processing()")
        self.update_processing()

    def add_checkbox_item(self, contour_id, text):
        """
        為列表項目動態加入一個 Checkbox 和標籤，並將狀態存入狀態表。
        """
        item_widget = QWidget(self)
        layout = QHBoxLayout()

        # 創建 Checkbox（不設置文字）
        checkbox = QCheckBox()
        checkbox.setChecked(self.checkbox_states.get(contour_id, True))  # 預設為 True
        checkbox.stateChanged.connect(lambda state: self.update_checkbox_state(contour_id, state))
        checkbox.setFixedSize(30, 16)  # 強制縮小 Checkbox 的尺寸
        # checkbox.setStyleSheet("""
        #     QCheckBox::indicator {
        #         width: 16px;
        #         height: 16px;
        #     }
        #     QCheckBox {
        #         spacing: 2px;  /* 控制 Checkbox 圖標和文字的距離 */
        #     }
        # """)

        # 使用 QLabel 單獨顯示文字
        label = QLabel(text)
        label.setStyleSheet("margin: 0px; padding: 0px;")  # 移除內外邊距
        label.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)  # 壓縮寬度占用

        # 添加到布局
        layout.addWidget(checkbox)
        layout.addWidget(label)

        # 調整布局的間距
        layout.setContentsMargins(0, 0, 0, 0)  # 移除內邊距
        layout.setSpacing(2)  # 減少 Checkbox 與文字的距離

        # 設置布局到 item_widget
        item_widget.setLayout(layout)

        # 添加到 QListWidget
        list_item = QListWidgetItem()
        self.contour_list.addItem(list_item)
        list_item.setSizeHint(QSize(0, 25))  # 手動設置寬高，高度設置為 25

        # 綁定 Checkbox 與 Widget
        self.contour_list.setItemWidget(list_item, item_widget)

        # 儲存 Checkbox 對應到 contour_id
        self.contour_checkboxes[contour_id] = checkbox
        return checkbox

    def update_contour_list(self, contours, hierarchy):
        """更新輪廓列表，確保 `QCheckBox` 正確顯示"""
        print("[DEBUG] update_contour_list() called")

        if not contours or hierarchy is None:
            print("[DEBUG] No contours found, clearing contour_list")
            self.contour_list.clear()
            return

        self.contour_list.clear()  # 清空列表
        hierarchy = hierarchy[0]

        contour_dict = {}
        for i, h in enumerate(hierarchy):
            parent = h[3]
            if parent == -1:
                contour_dict[i] = []
                child = h[2]
                while child != -1:
                    contour_dict[i].append(child)
                    child = hierarchy[child][0]

        for outer, holes in contour_dict.items():
            text = f"Contour = {outer}, Holes = [{', '.join(map(str, holes))}]" if holes else f"Contour = {outer}"
            
            print(f"[DEBUG] Creating item for: '{text}'")

            # **使用 `add_checkbox_item()` 來確保 `QCheckBox` 仍然存在**
            checkbox = self.add_checkbox_item(outer, text)
            checkbox.setChecked(self.checkbox_states.get(outer, True))

        print(f"[DEBUG] Contour list count after update: {self.contour_list.count()}")

        # **確認 `QCheckBox` 是否正常添加**
        for i in range(self.contour_list.count()):
            item = self.contour_list.item(i)
            widget = self.contour_list.itemWidget(item)
            if widget:
                label = widget.layout().itemAt(1).widget()  # 取得 QLabel
                print(f"[DEBUG] Contour list item {i}: '{label.text()}' (Widget: {widget})")
            else:
                print(f"[DEBUG] Contour list item {i}: '{item.text()}' (No Widget)")

    def update_processing(self):
        print("[DEBUG] update_processing() called")

        gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, self.threshold_value, 255, cv2.THRESH_BINARY_INV)
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

        if not contours or hierarchy is None:
            print("[DEBUG] No contours found.")
            self.contour_list.clear()
            self.processed_image = self.original_image.copy()
            self.update_display()
            return

        print(f"[DEBUG] Found {len(contours)} contours")
        self.update_contour_list(contours, hierarchy)

        output_image = self.original_image.copy()

        # **確保 `hierarchy` 只有一個層級**
        hierarchy = hierarchy[0]

        contour_hierarchy_map = {}  # {外輪廓: [內洞列表]}
        for i, h in enumerate(hierarchy):
            parent = h[3]
            if parent == -1:
                contour_hierarchy_map[i] = []
                child = h[2]
                while child != -1:
                    contour_hierarchy_map[i].append(child)
                    child = hierarchy[child][0]

        # **根據 Checkbox 狀態決定哪些輪廓要顯示**
        for outer, holes in contour_hierarchy_map.items():
            if outer in self.checkbox_states and not self.checkbox_states[outer]:
                print(f"[DEBUG] Skipping Contour {outer}")
                continue

            cv2.polylines(output_image, [contours[outer]], isClosed=True, color=(0, 255, 0), thickness=2)

            for hole in holes:
                cv2.polylines(output_image, [contours[hole]], isClosed=True, color=(0, 0, 255), thickness=2)

        # **高亮目前選中的輪廓**
        if hasattr(self, "selected_contour_id"):
            highlight_id = self.selected_contour_id
            if 0 <= highlight_id < len(contours):
                print(f"[DEBUG] Highlighting Contour ID: {highlight_id}")
                cv2.polylines(output_image, [contours[highlight_id]], isClosed=True, color=(0, 255, 255), thickness=3)

        self.processed_image = output_image
        self.update_display()

    def update_display(self):
        rgb_image = cv2.cvtColor(self.processed_image, cv2.COLOR_BGR2RGB)
        height, width, channel = rgb_image.shape
        q_image = QImage(rgb_image.data, width, height, width * 3, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        self.image_label.setPixmap(pixmap)
        self.image_label.setScaledContents(True)

    def load_new_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Load Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp)")
        if file_path:
            self.original_image = cv2.imread(file_path)
            self.update_processing()

    def save_results(self):
        cv2.imwrite("detected_polygons.jpg", self.processed_image)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PolygonSimplifierApp("input.png")
    window.show()
    sys.exit(app.exec_())
