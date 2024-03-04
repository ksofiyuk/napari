from napari.components.overlays.base import SceneOverlay


class LabelsBoundingBoxesOverlay(SceneOverlay):
    active: bool = False
    enabled: bool = True
    bounding_boxes: list = []

    _vispy_overlay = None

    def remove_selected_bounding_box(self):
        if self._vispy_overlay is not None:
            self._vispy_overlay.remove_selected_bounding_box()

    def undo(self):
        self._vispy_overlay.undo()

    def redo(self):
        self._vispy_overlay.redo()
