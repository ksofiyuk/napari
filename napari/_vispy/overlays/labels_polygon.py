import numpy as np
from vispy.scene.visuals import Compound, Line, Markers, Polygon

from napari._vispy.overlays.base import LayerOverlayMixin, VispySceneOverlay


class VispyLabelsPolygonOverlay(LayerOverlayMixin, VispySceneOverlay):
    def __init__(self, *, layer, overlay, parent=None):
        points = [(0, 0), (1, 1)]

        self._nodes_kwargs = {
            'face_color': (1, 1, 1, 0.5),
            'size': 8.0,
            'edge_width': 0.0,
            'edge_color': (0, 0, 0, 0.5),
        }

        self._nodes = Markers(pos=np.array(points), **self._nodes_kwargs)

        self._polygon = Polygon(
            pos=points,
            border_method='agg',
        )

        self._line = Line(pos=points, method='agg')

        super().__init__(
            node=Compound([self._polygon, self._nodes, self._line]),
            layer=layer,
            overlay=overlay,
            parent=parent,
        )
        self.overlay.events.points.connect(self._on_points_change)
        self.overlay.events.color.connect(self._on_color_change)
        self.overlay.events.dims_order.connect(self._on_points_change)

        self.reset()

    def _on_points_change(self):
        num_points = len(self.overlay.points)
        points = np.array(self.overlay.points).reshape((-1, 2))
        points = points[:, self.overlay.dims_order]

        if num_points > 2:
            self._polygon.visible = True
            self._line.visible = False
            self._polygon.pos = points
        else:
            self._polygon.visible = False
            self._line.visible = num_points == 2
            if self._line.visible:
                self._line.set_data(pos=points)

        self._nodes.set_data(
            pos=points,
            **self._nodes_kwargs,
        )

    def _on_color_change(self):
        border_color = self.overlay.color[:3] + (1,)  # always opaque

        # Workaround for VisPy's polygon bug: if you set opacity to exactly 0,
        # it keeps the previous visualization of the polygon without cleaning it
        polygon_color = self.overlay.color[:3] + (
            max(self.overlay.color[3], 1e-3),
        )

        self._polygon.color = polygon_color
        self._polygon.border_color = border_color
        self._line.set_data(color=border_color)

    def reset(self):
        super().reset()
        self._on_points_change()
        self._on_color_change()
