import sys
import cv2
import numpy as np

# from OpenGL.GL import glClear, GL_COLOR_BUFFER_BIT
# from OpenGL.GLUT import glutInit
from PyQt5.QtWidgets import QMessageBox

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QVBoxLayout, QHBoxLayout,
    QPushButton, QWidget, QCheckBox, QGroupBox, QFileDialog, QSlider, QListWidget, QListWidgetItem,
    QSizePolicy, QSpinBox
)

from PyQt5.QtCore import Qt, QSize, QTimer
from PyQt5.QtGui import QPixmap, QImage
# from scipy.spatial import Delaunay  # **三角剖分**

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
        self.min_contour_area = 100
        self.invalid_contours = set()  # **存放過小輪廓的 ID**

        # 儲存 Checkbox 狀態
        self.contour_checkboxes = {}
        self.checkbox_states = {}  # 儲存 Checkbox 的狀態，格式為 {contour_id: True/False}

        # 初始化 UI
        self.init_ui()
        self.update_processing()

    def init_ui(self):
        self.setWindowTitle("Polygon Detection and Filling")
        self.setGeometry(100, 100, 1400, 800)
        self.setMinimumSize(1400, 800)

        # **左側圖片顯示區**
        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setScaledContents(True)

        # **右側下半部：輪廓列表**
        self.contour_list = QListWidget(self)
        self.contour_list.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.contour_list.currentRowChanged.connect(self.on_row_changed)

        # **右側上半部：參數調整區**
        sliders_layout = QVBoxLayout()
        sliders_layout.addWidget(self.create_slider("Threshold", 0, 255, self.threshold_value, self.update_processing))

        # **Checkbox：是否顯示頂點**
        self.vertex_checkbox = QCheckBox("Show Vertices")
        self.vertex_checkbox.setChecked(self.show_vertices)
        self.vertex_checkbox.stateChanged.connect(self.update_processing)
        sliders_layout.addWidget(self.vertex_checkbox)

        # **Extrude 設定區域**
        extrude_layout = QHBoxLayout()

        # **Z 高度輸入框**
        self.extrude_height_input = QSpinBox()
        self.extrude_height_input.setRange(100, 1000)  # 設定範圍
        self.extrude_height_input.setValue(100)  # 預設值

        # **Extrude 按鈕**
        self.extrude_button = QPushButton("Extrude", self)
        self.extrude_button.clicked.connect(self.extrude_contour)

        # **將高度輸入框與按鈕放在同一行**
        extrude_layout.addWidget(self.extrude_height_input)
        extrude_layout.addWidget(self.extrude_button)

        # **封閉選項**
        self.fill_base_checkbox = QCheckBox("Fill Base (封底)")
        self.fill_base_checkbox.setChecked(True)  # 預設勾選

        self.fill_top_checkbox = QCheckBox("Fill Top (封頂)")
        self.fill_top_checkbox.setChecked(True)  # 預設勾選

        # **加入參數區**
        sliders_layout.addLayout(extrude_layout)
        sliders_layout.addWidget(self.fill_base_checkbox)
        sliders_layout.addWidget(self.fill_top_checkbox)

        # **按鈕區域**
        buttons_layout = QVBoxLayout()
        buttons_layout.addWidget(QPushButton("Load Image", self, clicked=self.load_new_image))
        buttons_layout.addWidget(QPushButton("Save", self, clicked=self.save_results))
        buttons_layout.addWidget(QPushButton("Quit", self, clicked=self.close))

        sliders_layout.addLayout(buttons_layout)

        # **右側區域（參數調整 + 輪廓列表）**
        right_layout = QVBoxLayout()
        sliders_group = QGroupBox("Parameters")
        sliders_group.setLayout(sliders_layout)
        right_layout.addWidget(sliders_group, stretch=1)  # 上半部參數調整區
        right_layout.addWidget(self.contour_list, stretch=2)  # 下半部列表區

        # **主佈局（左側圖片 + 右側控件）**
        main_layout = QHBoxLayout()
        main_layout.addWidget(self.image_label, stretch=3)  # 左側圖片區
        main_layout.addLayout(right_layout, stretch=2)  # 右側控制區

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

    def save_stl(self, faces, filename):
        """將三角形面片寫入 STL 檔案"""
        with open(filename, "w") as f:
            f.write("solid extruded_mesh\n")

            for tri in faces:
                f.write("  facet normal 0.0 0.0 0.0\n")
                f.write("    outer loop\n")
                for vertex in tri:
                    f.write(f"      vertex {vertex[0]} {vertex[1]} {vertex[2]}\n")
                f.write("    endloop\n")
                f.write("  endfacet\n")

            f.write("endsolid extruded_mesh\n")

    def create_cube(self, x, y, size, height):
        """建立一個立方體（修正三角形方向）"""
        p0 = (x, y, 0)
        p1 = (x + size, y, 0)
        p2 = (x + size, y + size, 0)
        p3 = (x, y + size, 0)
        
        p4 = (x, y, height)
        p5 = (x + size, y, height)
        p6 = (x + size, y + size, height)
        p7 = (x, y + size, height)

        # **修正所有面為逆時針順序 (CCW)**
        faces = [
            [p0, p2, p1], [p0, p3, p2],  # **底部 (XY 平面)**
            [p4, p5, p6], [p4, p6, p7],  # **頂部 (XY 平面)**
            [p0, p1, p5], [p0, p5, p4],  # **前面**
            [p1, p2, p6], [p1, p6, p5],  # **右側**
            [p2, p3, p7], [p2, p7, p6],  # **後面**
            [p3, p0, p4], [p3, p4, p7]   # **左側**
        ]
        return faces
    
    def extrude_single_contour(self, contour_id, contours, hierarchy, z_height, grid_size):
        """拉伸單個輪廓（包含內孔洞），回傳 STL 面片"""
        outer_contour = contours[contour_id]  # **外輪廓**
        hole_contours = [contours[i] for i, h in enumerate(hierarchy) if h[3] == contour_id]  # **內孔洞**

        print(f"[DEBUG] Extruding Contour ID: {contour_id}, Found {len(hole_contours)} holes")

        # **計算棋盤範圍**
        x_min, y_min = np.min(outer_contour[:, 0, :], axis=0)
        x_max, y_max = np.max(outer_contour[:, 0, :], axis=0)

        print(f"[DEBUG] Grid Boundary: X=({x_min}, {x_max}), Y=({y_min}, {y_max})")

        faces = []  # **儲存 STL 面片**

        # **遍歷棋盤格，判斷哪些方格在輪廓內**
        for x in range(x_min, x_max, grid_size):
            for y in range(y_min, y_max, grid_size):
                cell_center = (x + grid_size / 2, y + grid_size / 2)

                # **檢查是否在外輪廓內**
                inside_outer = cv2.pointPolygonTest(outer_contour, cell_center, measureDist=False) >= 0
                inside_hole = any(cv2.pointPolygonTest(h, cell_center, measureDist=False) >= 0 for h in hole_contours)

                if inside_outer and not inside_hole:
                    cube_faces = self.create_cube(x, y, grid_size, z_height)
                    faces.extend(cube_faces)

        return faces

    def extrude_contour(self):
        """拉伸所有勾選的輪廓，生成 STL"""
        z_height = self.extrude_height_input.value()
        grid_size = 10  # 棋盤格大小

        print(f"[DEBUG] Extruding all checked contours, Height: {z_height}, Grid Size: {grid_size}")

        # **確保 contours 存在**
        if not self.contours:
            print("[DEBUG] No contours available, skipping extrusion.")
            return

        faces = []

        # **遍歷所有勾選的輪廓**
        for contour_id in self.checkbox_states:
            if self.checkbox_states[contour_id]:  # **只處理勾選的輪廓**
                if contour_id >= len(self.contours):  # **檢查是否超出範圍**
                    print(f"[DEBUG] Skipping invalid contour ID: {contour_id}")
                    continue

                faces.extend(self.extrude_single_contour(contour_id, self.contours, self.hierarchy, z_height, grid_size))

        # **儲存 STL**
        self.save_stl(faces, "extruded_voxel_mesh_all.stl")
        print("[DEBUG] STL file saved: extruded_voxel_mesh_all.stl")
            
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
    
    def calculate_contour_area(self, contour_id, contours, hierarchy):
        """
        計算指定輪廓 ID 的面積，考慮內洞的影響
        :param contour_id: 外輪廓 ID
        :param contours: 所有輪廓列表
        :param hierarchy: 層級結構
        :return: 計算後的有效面積（內洞會被扣除）
        """
        # hierarchy = hierarchy[0]  # 提取層級資訊
        total_area = cv2.contourArea(contours[contour_id])  # 先計算外輪廓面積

        # **扣除內洞面積**
        for i, h in enumerate(hierarchy):
            if h[3] == contour_id:  # **找到屬於這個外輪廓的內洞**
                total_area -= cv2.contourArea(contours[i])

        return total_area

    def update_contour_list(self, contours, hierarchy):
        """更新輪廓列表，確保 `QCheckBox` 正確顯示"""
        print("[DEBUG] update_contour_list() called")

        if not contours or hierarchy is None:
            print("[DEBUG] No contours found, clearing contour_list")
            self.contour_list.clear()
            return

        # **清空之前的 invalid_contours**
        self.invalid_contours.clear()
        self.contour_list.clear()  # 清空列表
        hierarchy = hierarchy[0]

        contour_dict = {}
        for i, h in enumerate(hierarchy):
            parent = h[3]
            if parent == -1:
                area = self.calculate_contour_area(i, contours, hierarchy)
                if area >= self.min_contour_area:  # **只加入足夠大的輪廓**
                    contour_dict[i] = []
                    child = h[2]
                    while child != -1:
                        contour_dict[i].append(child)
                        child = hierarchy[child][0]
                else:
                    self.invalid_contours.add(i)

        for outer, holes in contour_dict.items():
            text = f"Contour = {outer}, Holes = [{', '.join(map(str, holes))}]" if holes else f"Contour = {outer}"
            checkbox = self.add_checkbox_item(outer, text)

            # **確保 checkbox 狀態被正確存入**
            self.checkbox_states[outer] = checkbox.isChecked()  # **這行讓未點擊的也會有狀態**

        print("[DEBUG] Updated checkbox_states =", self.checkbox_states)

        # for outer, holes in contour_dict.items():
        #     text = f"Contour = {outer}, Holes = [{', '.join(map(str, holes))}]" if holes else f"Contour = {outer}"
            
        #     print(f"[DEBUG] Creating item for: '{text}'")

        #     # **使用 `add_checkbox_item()` 來確保 `QCheckBox` 仍然存在**
        #     checkbox = self.add_checkbox_item(outer, text)
        #     checkbox.setChecked(self.checkbox_states.get(outer, True))

        # print(f"[DEBUG] Contour list count after update: {self.contour_list.count()}")

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
        self.contours, _hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

        if not self.contours or _hierarchy is None:
            print("[DEBUG] No contours found.")
            self.contour_list.clear()
            self.processed_image = self.original_image.copy()
            self.update_display()
            return

        print(f"[DEBUG] Found {len(self.contours)} contours")
        self.update_contour_list(self.contours, _hierarchy)

        output_image = self.original_image.copy()
        self.hierarchy = _hierarchy[0]  # 確保 `hierarchy` 正確展開

        # **建立輪廓與內洞對應關係**
        contour_hierarchy_map = {}  # {外輪廓: [內洞列表]}
        hole_to_parent = {}  # {內洞: 外輪廓}

        for i, h in enumerate(self.hierarchy):

            if i in self.invalid_contours:
                print(f"[DEBUG] Skipping small contour ID {i}")
                continue  # **跳過小輪廓**

            parent = h[3]
            if parent == -1:
                contour_hierarchy_map[i] = []
                child = h[2]
                while child != -1:
                    contour_hierarchy_map[i].append(child)
                    hole_to_parent[child] = i
                    child = self.hierarchy[child][0]

        # **根據 Checkbox 狀態決定哪些輪廓要顯示**
        for outer, holes in contour_hierarchy_map.items():
            if outer in self.checkbox_states and not self.checkbox_states[outer]:
                print(f"[DEBUG] Skipping Contour {outer} (Checkbox Unchecked)")
                continue  # 不畫出該外輪廓

            # 畫外輪廓
            cv2.polylines(output_image, [self.contours[outer]], isClosed=True, color=(0, 255, 0), thickness=2)

            # 畫內洞
            for hole in holes:
                if hole in self.checkbox_states and not self.checkbox_states[hole]:
                    print(f"[DEBUG] Skipping Hole {hole} (Checkbox Unchecked)")
                    continue  # 不畫出該內洞
                cv2.polylines(output_image, [self.contours[hole]], isClosed=True, color=(0, 0, 255), thickness=2)

        # **高亮顯示完整輪廓組（包括 contour 和 holes）**
        if hasattr(self, "selected_contour_id"):
            highlight_id = self.selected_contour_id
            highlight_group = set()

            # **如果選中的是外輪廓，包含其所有內洞**
            if highlight_id in contour_hierarchy_map and self.checkbox_states.get(highlight_id, True):
                highlight_group.add(highlight_id)
                highlight_group.update(contour_hierarchy_map[highlight_id])

            # **如果選中的是內洞，回溯到外輪廓並包含整組**
            # elif highlight_id in hole_to_parent:
            #     parent_id = hole_to_parent[highlight_id]
            #     highlight_group.add(parent_id)
            #     highlight_group.update(contour_hierarchy_map[parent_id])

            # **過濾掉 Checkbox 取消選取的輪廓**
            # highlight_group = {cid for cid in highlight_group if self.checkbox_states.get(cid, True)}

            if highlight_group:
                print(f"[DEBUG] Highlighting Contour Group: {highlight_group}")
                for cid in highlight_group:
                    cv2.polylines(output_image, [self.contours[cid]], isClosed=True, color=(0, 255, 255), thickness=3)
            else:
                print("[DEBUG] No contours to highlight (All Checkboxes Unchecked)")

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
