import numpy as np
from qtpy.QtCore import Qt
from qtpy.QtGui import QColor, QIcon, QPainter, QPixmap
from qtpy.QtWidgets import (
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QWidget,
)
from superqt import QLargeIntSpinBox

from napari._qt.dialogs.qt_modal import QtPopup
from napari._qt.layer_controls.widgets.qt_widget_controls_base import (
    QtWidgetControlsBase,
    QtWrappedLabel,
)
from napari._qt.utils import qt_signals_blocked
from napari._qt.widgets.qt_color_swatch import QColorSwatchEdit
from napari.layers import Labels
from napari.layers.labels._labels_key_bindings import new_label
from napari.layers.labels._labels_utils import get_dtype
from napari.utils._dtype import get_dtype_limits


def paint_checkerboard(painter: QPainter, height: int) -> None:
    for i in range(height // 4):
        for j in range(height // 4):
            if (i % 2 == 0 and j % 2 == 0) or (i % 2 == 1 and j % 2 == 1):
                painter.setPen(QColor(230, 230, 230))
                painter.setBrush(QColor(230, 230, 230))
            else:
                painter.setPen(QColor(25, 25, 25))
                painter.setBrush(QColor(25, 25, 25))
            painter.drawRect(i * 4, j * 4, 5, 5)


class QtColorBox(QWidget):
    """A widget that shows a square with the current label color.

    Parameters
    ----------
    size : int
        A size of the color box.
    """

    def __init__(self, size: int = 24) -> None:
        super().__init__()

        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self._height = size
        self.setFixedWidth(self._height)
        self.setFixedHeight(self._height)
        self.setToolTip('Selected label')

        self._color = None

    def set_color(self, color):
        self._color = color
        self.update()

    def paintEvent(self, event):
        """Paint the colorbox.  If no color, display a checkerboard pattern.

        Parameters
        ----------
        event : qtpy.QtCore.QEvent
            Event from the Qt context.
        """
        painter = QPainter(self)
        if self._color is None:
            paint_checkerboard(painter, self._height)
        else:
            color = np.round(255 * self._color).astype(int)
            painter.setPen(QColor(*list(color)))
            painter.setBrush(QColor(*list(color)))
            painter.drawRect(0, 0, self._height, self._height)


class QtLabelSpinBox(QWidget):
    def __init__(self, layer) -> None:
        super().__init__()
        self.layer = layer
        self.color_swatch = QtColorBox()

        self.spinbox = QLargeIntSpinBox()
        self.spinbox.setKeyboardTracking(False)
        self.spinbox.setAlignment(Qt.AlignmentFlag.AlignCenter)

        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.color_swatch, 0)
        layout.addWidget(self.spinbox, 1)
        self.setLayout(layout)

        self.spinbox.valueChanged.connect(self.update_selected)

        self.layer.events.selected_label.connect(self._on_selection_change)
        self.layer.events.data.connect(self._on_data_change)
        self.layer.events.colormap.connect(self._on_color_change)
        self.layer.events.selected_label.connect(self._on_selection_change)

        self._on_data_change()
        self._on_color_change()
        self._on_selection_change()

    def _on_color_change(self) -> None:
        color = self.layer.get_color(self.layer.selected_label)
        with qt_signals_blocked(self.color_swatch):
            self.color_swatch.set_color(color)

    def _on_selection_change(self) -> None:
        with qt_signals_blocked(self.spinbox):
            self.spinbox.setValue(self.layer.selected_label)

    def update_selected(self, value: int) -> None:
        self.layer.selected_label = value
        self.spinbox.clearFocus()
        # TODO: decouple
        self.parent().setFocus()

    def _on_data_change(self) -> None:
        with qt_signals_blocked(self.spinbox):
            dtype_lims = get_dtype_limits(get_dtype(self.layer))
            self.spinbox.setRange(*dtype_lims)


class QtPredefinedLabelCombobox(QComboBox):
    def __init__(self, layer, size: int = 24) -> None:
        super().__init__()
        self.layer = layer

        self._height = size
        self.setFixedHeight(self._height)
        self.setToolTip('Selected label')

        self._current_elements = []

        self.currentTextChanged.connect(self.update_selected)

        self.layer.events.selected_label.connect(self._on_selection_change)
        self.layer.events.predefined_labels.connect(
            self._on_predefined_labels_change
        )
        self.layer.events.colormap.connect(self._on_color_change)

        self._on_color_change()

    def _on_selection_change(self):
        selected = self.layer.selected_label
        predefined = self.layer.predefined_labels
        if predefined is None:
            return

        if selected not in self._current_elements:
            # layer.predefined_labels auto-includes new selected labels that were
            # previously undefined, so we just need to rebuild
            self._on_predefined_labels_change()

        item_index = self._current_elements.index(selected)
        with qt_signals_blocked(self):
            self.setCurrentIndex(item_index)

    def update_selected(self):
        if self.layer.predefined_labels is not None:
            self.layer.selected_label = self._current_elements[
                self.currentIndex()
            ]

    def _on_color_change(self):
        self._on_predefined_labels_change()
        self._on_selection_change()

    def _on_predefined_labels_change(self):
        self._current_elements = []
        if self.layer.predefined_labels is None:
            return

        labels = self.layer.predefined_labels

        with qt_signals_blocked(self):
            for i, (label, name) in enumerate(labels.items()):
                self._current_elements.append(label)
                if i >= self.count():
                    self.addItem('')

                color = self.layer.get_color(label)
                color_pixmap = QPixmap(self._height, self._height)

                if color is None:
                    paint_checkerboard(QPainter(color_pixmap), self._height)
                else:
                    color = np.round(255 * color[:3]).astype(int)
                    color_pixmap.fill(QColor(*color.tolist()))

                color_icon = QIcon(color_pixmap)
                item_text = str(label) + (': ' + name if name else '')

                self.setItemIcon(i, color_icon)
                self.setItemText(i, item_text)

            for _ in range(self.count() - len(labels)):
                self.removeItem(self.count() - 1)


class QNewNamedLabelDialog(QtPopup):
    def __init__(self, *args, layer, **kwargs):
        super().__init__(*args, **kwargs)
        self.layer = layer
        self.name_edit = QLineEdit()
        ok_button = QPushButton('OK')
        self.name_edit.returnPressed.connect(ok_button.click)
        ok_button.clicked.connect(self.add_label)
        self.color_edit = QColorSwatchEdit(initial_color=np.random.rand(3))

        layout = QFormLayout()
        layout.addRow('label name:', self.name_edit)
        layout.addRow('label color:', self.color_edit)
        layout.addRow(ok_button)
        self.frame.setLayout(layout)

    def add_label(self):
        new_name = self.name_edit.text()
        predefined_labels = self.layer.predefined_labels.copy()
        if not new_name:
            return
        if new_name in predefined_labels:
            raise ValueError(f'"{new_name}" is already in predefined_labels')
        next_unused = self.layer.next_unused()
        self.layer.colormap.color_dict[next_unused] = self.color_edit.color
        predefined_labels[next_unused] = new_name
        self.layer.predefined_labels = predefined_labels
        self.layer.selected_label = next_unused

        self.close()


class QtCurrentLabelControl(QtWidgetControlsBase):
    """
    Class that wraps the connection of events/signals between the current label
    layer attribute and Qt widgets.

    Parameters
    ----------
    parent: qtpy.QtWidgets.QWidget
        An instance of QWidget that will be used as widgets parent
    layer : napari.layers.Labels
        An instance of a napari Labels layer.

    Attributes
    ----------
    colorbox : QtColorBox
        Widget that shows current layer label color.
    label_color : qtpy.QtWidget.QWidget
        Wrapper widget for the selection_spinbox and colorbox widgets.
    label_color_label : napari._qt.layer_controls.widgets.qt_widget_controls_base.QtWrappedLabel
        Label for the label chooser widget.
    selection_spinbox : superqt.QLargeIntSpinBox
        Widget to select a specific label by its index.
        N.B. cannot represent labels > 2**53.
    new_label_button : qtpy.QtWidgets.QPushButton
        Button to add a new label to the label layer.
    """

    _layer: Labels

    def __init__(self, parent: QWidget, layer: Labels) -> None:
        super().__init__(parent, layer)

        self.selection_spinbox = QtLabelSpinBox(layer)
        self.selection_combobox = QtPredefinedLabelCombobox(layer)

        self.new_label_button = QPushButton()
        self.new_label_button.setText('new')
        self.new_label_button.clicked.connect(self._on_new_button_click)

        self.current_label_label = QtWrappedLabel('label:')
        self.current_label_row = QWidget()

        color_layout = QHBoxLayout()
        color_layout.setContentsMargins(0, 0, 0, 0)
        color_layout.setSpacing(4)
        color_layout.addWidget(self.selection_spinbox, 1)
        color_layout.addWidget(self.selection_combobox, 1)
        color_layout.addWidget(self.new_label_button, 0)

        self.current_label_row.setLayout(color_layout)
        self.current_label_row.setProperty('foreground', 'true')

        self._layer.events.predefined_labels.connect(
            self._on_predefined_labels_change
        )
        self._on_predefined_labels_change()

    def _on_predefined_labels_change(self) -> None:
        if self._layer.predefined_labels is None:
            self.selection_combobox.setVisible(False)
            self.selection_spinbox.setVisible(True)
        else:
            self.selection_spinbox.setVisible(False)
            self.selection_combobox.setVisible(True)

    def _on_new_button_click(self):
        """Select a new label for the labels layer when the button is clicked."""
        if self._layer.predefined_labels is None:
            new_label(self._layer)
        else:
            diag = QNewNamedLabelDialog(
                parent=self.new_label_button, layer=self._layer
            )
            diag.show_right_of_mouse()

    def get_widget_controls(self) -> list[tuple[QtWrappedLabel, QWidget]]:
        return [(self.current_label_label, self.current_label_row)]
