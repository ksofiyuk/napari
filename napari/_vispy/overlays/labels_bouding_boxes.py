from enum import auto
from typing import Optional, Tuple

import numpy as np
import numpy.typing as npt
from vispy.scene.visuals import Compound, Markers, Rectangle

from napari._vispy.overlays.base import LayerOverlayMixin, VispySceneOverlay
from napari.components.overlays import LabelsBoundingBoxesOverlay
from napari.layers import Labels
from napari.layers.labels._labels_utils import mouse_event_to_labels_coordinate
from napari.utils.misc import StringEnum


class InteractionState(StringEnum):
    IDLE = auto()
    BBOX_CREATING = auto()
    BBOX_MODIFYING = auto()
    BBOX_SELECTED = auto()


class Operation(StringEnum):
    CREATE_BB = auto()
    REMOVE_BB = auto()
    CHANGE_BB = auto()


class VispyLabelsBoundingBoxesOverlay(LayerOverlayMixin, VispySceneOverlay):
    def __init__(
        self,
        *,
        layer: Labels,
        overlay: LabelsBoundingBoxesOverlay,
        parent=None,
    ):
        self._nodes_kwargs = {
            'face_color': (1, 1, 1, 1),
            'size': 7.0,
            'edge_width': 2.0,
            'symbol': 'square',
        }

        self._drag_nodes = Markers(pos=np.empty((0, 2)), **self._nodes_kwargs)
        self._bounding_boxes = Compound([])
        self._state: InteractionState = InteractionState.IDLE
        self._selected_bbox: Optional[RectangleWithLabel] = None
        self._label_before_selection: Optional[int] = None
        self._undo_history = []
        self._redo_history = []

        super().__init__(
            node=Compound([self._bounding_boxes, self._drag_nodes]),
            layer=layer,
            overlay=overlay,
            parent=parent,
        )

        self.overlay.events.enabled.connect(self._on_enabled_change)
        self.overlay.events.bounding_boxes.connect(
            self._on_bounding_boxes_change
        )
        self.layer.mouse_drag_callbacks.append(self._on_mouse_press_and_drag)

        layer.events.selected_label.connect(self._on_selected_label_change)
        layer.events.colormap.connect(self._update_color)
        layer.events.color_mode.connect(self._update_color)
        layer.events.opacity.connect(self._update_opacity)

        self.reset()
        self._update_color()
        # If there are no points, it won't be visible
        self.overlay.visible = True
        self.overlay._vispy_overlay = self

    def _update_opacity(self):
        for subvisual in self._bounding_boxes._subvisuals:
            subvisual.color.alpha = self.layer.opacity
            subvisual.color = subvisual.color

    def _on_selected_label_change(self):
        if self._state == InteractionState.BBOX_SELECTED:
            if (
                self.layer.selected_label != 0
                and self.layer.selected_label != self._selected_bbox.label
            ):
                self._change_bounding_box(
                    self._selected_bbox, {"label": self.layer.selected_label}
                )
        else:
            self._update_color()

    def _on_enabled_change(self):
        if not self.overlay.enabled:
            self._quit_selection_mode()

    def _on_bounding_boxes_change(self):
        self._quit_selection_mode()
        self._undo_history = []
        self._redo_history = []

        current_rects = self._bounding_boxes._subvisuals[:]
        for x in current_rects:
            self._bounding_boxes.remove_subvisual(x)

        for bounding_box_dict in self.overlay.bounding_boxes:
            rect_visual = self._create_rect(
                center=np.array(
                    (bounding_box_dict["yc"], bounding_box_dict["xc"])
                ),
                height=bounding_box_dict["h"],
                width=bounding_box_dict["w"],
                label=bounding_box_dict["label"],
            )
            self._bounding_boxes.add_subvisual(rect_visual)

    def _update_color(self):
        for rect_visual in self._bounding_boxes._subvisuals:
            self._redraw_bbox_color(rect_visual)

    def _redraw_bbox_color(self, rect_visual):
        color, border_color = self._get_bbox_color(rect_visual.label)
        rect_visual.color = color
        rect_visual.border_color = border_color

    def _get_bbox_color(self, label):
        rect_label_color = self.layer.get_color(label).tolist()
        color = rect_label_color[:3] + [self.layer.opacity]
        border_color = rect_label_color[:3]
        return color, border_color

    def _on_mouse_press_and_drag(self, layer, event):
        if not self.overlay.enabled:
            return

        if event.button == 1:
            press_pos = self._get_mouse_coordinates(event)
            yield

            if self._state == InteractionState.IDLE:
                if event.type == 'mouse_move':
                    for _ in self._create_new_bounding_box(event, press_pos):
                        yield
                else:
                    self._find_and_select_bounding_box(press_pos)
            elif self._state == InteractionState.BBOX_SELECTED:
                selected_modifier = self._find_drag_modifier(press_pos)

                if selected_modifier is not None:
                    for _ in self._drag_bbox_node(
                        event, press_pos, selected_modifier
                    ):
                        yield
                else:
                    if self._find_and_select_bounding_box(press_pos) == 0:
                        self._quit_selection_mode()

    def _create_new_bounding_box(self, event, start_pos):
        if self.layer.selected_label == self.layer._background_label:
            return

        self._state = InteractionState.BBOX_CREATING

        width, height = 0, 0

        rect_visual = self._create_rect(
            center=start_pos,
            height=1e-3,
            width=1e-3,
            label=self.layer.selected_label,
        )
        self._bounding_boxes.add_subvisual(rect_visual)
        self._bounding_boxes.update()

        while event.type == 'mouse_move':
            pos = self._get_mouse_coordinates(event)
            height, width = np.abs(pos - start_pos)
            if width > 0 and height > 0:
                rect_visual.set_rect(
                    self._dims_displayed,
                    0.5 * (start_pos + pos),
                    height,
                    width,
                )
            yield

        if width * height > 9:
            self._add_bounding_box(rect_visual, add_subvisual=False)
        else:
            self._bounding_boxes.remove_subvisual(rect_visual)

        self._state = InteractionState.IDLE

    def _create_rect(self, center, height, width, label):
        color, border_color = self._get_bbox_color(label)
        rect_visual = RectangleWithLabel(
            center=center,
            height=height,
            width=width,
            label=label,
            dims_displayed=self._dims_displayed,
            color=color,
            border_color=border_color,
            border_width=2,
        )

        return rect_visual

    def _find_and_select_bounding_box(self, click_pos) -> bool:
        matched_rect = None

        rect: RectangleWithLabel
        for rect in self._bounding_boxes._subvisuals:
            if rect.contains(click_pos) and (
                matched_rect is None or matched_rect.area > rect.area
            ):
                matched_rect = rect

        if matched_rect is None:
            return False

        if self._state == InteractionState.IDLE:
            self._label_before_selection = self.layer.selected_label
        self._state = InteractionState.BBOX_SELECTED
        self._selected_bbox = matched_rect
        self.layer.selected_label = matched_rect.label
        self._draw_selected_bbox_drag_nodes()

        return True

    def _draw_selected_bbox_drag_nodes(self):
        drag_points = self._selected_bbox.drag_modifiers[:, :2]

        edge_color = self._selected_bbox.border_color.darker(dv=0.5)
        self._drag_nodes.set_data(
            pos=self._to_displayed(drag_points),
            edge_color=edge_color,
            **self._nodes_kwargs,
        )

    def _find_drag_modifier(self, point):
        drag_modifiers = self._selected_bbox.drag_modifiers
        _, _, w, h = self._selected_bbox.xywh
        drag_radius = max(2, int(0.5 * (w + h) / 20)) ** 2
        selected_modifier = None
        last_r = None

        for i, (y, x) in enumerate(drag_modifiers[:, :2]):
            r = (y - point[0]) ** 2 + (x - point[1]) ** 2

            if r <= drag_radius and (last_r is None or last_r > r):
                selected_modifier = drag_modifiers[i]
                last_r = r

        return selected_modifier

    def _drag_bbox_node(self, event, start_point, node_modifier):
        self._state = InteractionState.BBOX_MODIFYING
        bbox = self._selected_bbox
        x, y, w, h = bbox.xywh
        y_mult, y_dir, x_mult, x_dir = node_modifier[2:]
        cx, cy = x + 0.5 * w, y + 0.5 * h
        initial_bbox_state = bbox.asdict()

        while event.type == 'mouse_move':
            pos = self._get_mouse_coordinates(event)
            dy, dx = pos - start_point

            new_cx = cx + 0.5 * dx * x_mult
            new_cy = cy + 0.5 * dy * y_mult
            new_w = max(abs(w + x_dir * dx), 1e-3)
            new_h = max(abs(h + y_dir * dy), 1e-3)

            bbox.set_rect(
                self._dims_displayed, np.array((new_cy, new_cx)), new_h, new_w
            )
            self._draw_selected_bbox_drag_nodes()
            yield

        self._change_bounding_box(
            bbox, bbox.asdict(), prev_state=initial_bbox_state
        )

        self._state = InteractionState.BBOX_SELECTED

    def _change_bounding_box(
        self,
        bounding_box: 'RectangleWithLabel',
        new_state,
        prev_state: Optional = None,
        save_history: bool = True,
    ):
        if prev_state is None:
            prev_state = bounding_box.asdict()
        new_state = {
            key: value
            for key, value in new_state.items()
            if prev_state[key] != value
        }
        if not new_state:
            return

        bounding_box.update_state(new_state, self._dims_displayed)
        if "label" in new_state:
            self._redraw_bbox_color(bounding_box)

        if bounding_box is self._selected_bbox:
            self._draw_selected_bbox_drag_nodes()

        if save_history:
            self._add_operation_to_history(
                Operation.CHANGE_BB, (bounding_box, prev_state)
            )
        self._update_bounding_boxes()

    def remove_selected_bounding_box(self):
        if self._state == InteractionState.BBOX_SELECTED:
            self._remove_bounding_box(self._selected_bbox)

    def _remove_bounding_box(
        self, bounding_box: 'RectangleWithLabel', save_history: bool = True
    ) -> None:
        if bounding_box is self._selected_bbox:
            self._quit_selection_mode()
        self._bounding_boxes.remove_subvisual(bounding_box)
        if save_history:
            self._add_operation_to_history(Operation.REMOVE_BB, bounding_box)
        self._update_bounding_boxes()

    def _quit_selection_mode(self):
        if self._state != InteractionState.BBOX_SELECTED:
            return

        self._drag_nodes.set_data(pos=np.empty((0, 2)))
        self._selected_bbox = None
        self._state = InteractionState.IDLE
        self.layer.selected_label = self._label_before_selection

    def _add_bounding_box(
        self,
        bounding_box: 'RectangleWithLabel',
        save_history: bool = True,
        add_subvisual: bool = True,
    ) -> None:
        if add_subvisual:
            self._bounding_boxes.add_subvisual(bounding_box)
        if save_history:
            self._add_operation_to_history(Operation.CREATE_BB, bounding_box)
        self._update_bounding_boxes()

    def _add_operation_to_history(self, op_type: Operation, op_state):
        self._redo_history = []
        self._undo_history.append((op_type, op_state))

    def undo(self):
        self._revert_operation(undo=True)

    def redo(self):
        self._revert_operation(undo=False)

    def _revert_operation(self, undo: bool = True) -> None:
        ops_pool = self._undo_history if undo else self._redo_history

        if not ops_pool:
            return

        op_type, op_state = ops_pool.pop()
        if op_type == Operation.CHANGE_BB:
            bounding_box, prev_state = op_state
            revert_op = (op_type, (bounding_box, bounding_box.asdict()))

            self._change_bounding_box(
                bounding_box, prev_state, save_history=False
            )
        elif op_type == Operation.CREATE_BB:
            bounding_box = op_state
            revert_op = (Operation.REMOVE_BB, bounding_box)
            self._remove_bounding_box(bounding_box, save_history=False)
        elif op_type == Operation.REMOVE_BB:
            bounding_box = op_state
            revert_op = (Operation.CREATE_BB, bounding_box)
            self._add_bounding_box(bounding_box, save_history=False)
        else:
            raise NotImplementedError

        if undo:
            self._redo_history.append(revert_op)
        else:
            self._undo_history.append(revert_op)

    def _update_bounding_boxes(self):
        self.overlay.events.bounding_boxes.disconnect(
            self._on_bounding_boxes_change
        )
        new_bounding_boxes = [
            bb.asdict() for bb in self._bounding_boxes._subvisuals
        ]
        self.overlay.bounding_boxes = new_bounding_boxes
        self.overlay.events.bounding_boxes.connect(
            self._on_bounding_boxes_change
        )

    def _get_mouse_coordinates(self, event):
        pos = mouse_event_to_labels_coordinate(self.layer, event)
        if pos is None:
            return None

        pos = np.array(pos, dtype=float)
        pos[self._dims_displayed] += 0.5

        return pos

    def _to_displayed(self, point: npt.NDArray):
        if len(point.shape) == 2:
            return point[:, self._dims_displayed[::-1]]
        return point[self._dims_displayed[::-1]]

    @property
    def _dims_displayed(self):
        return self.layer._slice_input.displayed

    @property
    def _num_points(self):
        return len(self.overlay.points)

    def reset(self):
        super().reset()


class RectangleWithLabel(Rectangle):
    def __init__(
        self,
        center,
        width: float,
        height: float,
        label: int,
        dims_displayed,
        **kwargs,
    ):
        self.label = label
        self._orig_center = center  # (y_row, x_col)
        self._orig_hw = height, width

        if dims_displayed[0] > dims_displayed[1]:
            width, height = height, width

        super().__init__(
            center=center[dims_displayed[::-1]],
            width=width,
            height=height,
            **kwargs,
        )

    def set_rect(self, dims_displayed, center=None, height=None, width=None):
        if center is not None:
            self._orig_center = center
        if height is not None and width is not None:
            self._orig_hw = height, width

        self.center = self._orig_center[dims_displayed[::-1]]
        height, width = self._orig_hw
        if dims_displayed[0] > dims_displayed[1]:
            width, height = height, width
        self.width, self.height = width, height

    def contains(self, point) -> bool:
        y, x = point
        h, w = self._orig_hw
        cy, cx = self._orig_center

        if (cy - 0.5 * h <= y <= cy + 0.5 * h) and (
            cx - 0.5 * w <= x <= cx + 0.5 * w
        ):
            return True

        return False

    @property
    def xywh(self) -> Tuple[float, float, float, float]:
        h, w = self._orig_hw
        cy, cx = self._orig_center

        return cx - 0.5 * w, cy - 0.5 * h, w, h

    @property
    def area(self) -> float:
        return self._orig_hw[0] * self._orig_hw[1]

    @property
    def drag_modifiers(self) -> npt.NDArray:
        x, y, w, h = self.xywh
        # fmt: off
        corners = np.array([
            # Corners
            (y, x,          1, -1, 1, -1),  # top left
            (y, x + w,      1, -1, 1,  1),  # top right
            (y + h, x + w,  1,  1, 1,  1),  # bottom right
            (y + h, x,      1,  1, 1, -1),  # bottom left

            # Midpoints
            (y, x + 0.5 * w,     1, -1, 0, 0),  # top
            (y + 0.5 * h, x + w, 0, 0, 1, 1),   # right
            (y + h, x + 0.5 * w, 1, 1, 0, 0),   # bottom
            (y + 0.5 * h, x,     0, 0, 1, -1),  # left

            # Center
            (y + 0.5 * h, x + 0.5 * w, 2, 0, 2, 0),
        ])
        # fmt: on

        return corners

    def asdict(self):
        return {
            "xc": self._orig_center[1],
            "yc": self._orig_center[0],
            "w": self._orig_hw[1],
            "h": self._orig_hw[0],
            "label": self.label,
        }

    def update_state(self, new_state, dims_displayed):
        if (
            "yc" in new_state
            or "xc" in new_state
            or "h" in new_state
            or "w" in new_state
        ):
            self.set_rect(
                dims_displayed,
                center=np.array(
                    (
                        new_state.get("yc", self._orig_center[0]),
                        new_state.get("xc", self._orig_center[1]),
                    )
                ),
                height=new_state.get("h", self._orig_hw[0]),
                width=new_state.get("w", self._orig_hw[1]),
            )
        self.label = new_state.get("label", self.label)
