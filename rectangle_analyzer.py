from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Callable, Optional, cast

Number = float | int


@dataclass
class Rectangle:
    """A 2D axis-aligned rectangle."""

    x: Number
    y: Number
    width: Number
    height: Number

    def __post_init__(self):
        for name, value in vars(self).items():
            if not isinstance(value, (int, float)):
                raise TypeError(
                    f"{name} must be an int or float (got {type(value).__name__})"
                )
            setattr(self, name, float(value))

        if self.width <= 0 or self.height <= 0:
            raise ValueError(
                f"Invalid rectangle: width and height must be > 0, got width={self.width}, height={self.height}."
            )

    @classmethod
    def from_dicts(cls, rectangles: list[dict[str, Number]]) -> list[Rectangle]:
        """
        Create a list of Rectangle objects from a list of dictionaries.
        Args:
            rectangles: List of dicts with keys 'x', 'y', 'width', and 'height'.
        """

        return [cls(**rect) for rect in rectangles]


class RectangleAnalyzer:
    """Analyze geometric relationships between 2D axis-aligned rectangles."""

    def __init__(self, rectangles: list[dict[str, Number]]):
        """
        Initialize analyzer with list of rectangles.
        Each rectangle is a dict with keys: x, y, width, height.
        """

        self.rectangles: list[Rectangle] = Rectangle.from_dicts(rectangles)
        if not self.rectangles:
            raise ValueError("No rectangles provided.")

    @staticmethod
    def calculate_rectangle_area(rect: Rectangle) -> float:
        """
        Calculate the area of a given rectangle
        """
        return rect.width * rect.height

    @staticmethod
    def _get_rect_bounds(rect: Rectangle) -> tuple[float, float, float, float]:
        """
        Return the bounding coordinates (x1, y1, x2, y2) for a rectangle.
        """
        return (
            rect.x,
            rect.y,
            rect.x + rect.width,
            rect.y + rect.height,
        )

    @staticmethod
    def _compute_overlap(rects: list[Rectangle]) -> Optional[Rectangle]:
        """
        Compute the overlap region of multiple rectangles.

        Returns:
            A Rectangle instance corresponding to the overlapping rectangle region, or None if there is no overlap.
        """
        if not rects:
            return None

        # Extract all rectangle bounds
        bounds = [RectangleAnalyzer._get_rect_bounds(r) for r in rects]

        # Unpack all left/bottom/right/top coordinates separately
        a1_values = [b[0] for b in bounds]  # left x-axis edges (x1)
        b1_values = [b[1] for b in bounds]  # bottom y-axis edges (y1)
        a2_values = [b[2] for b in bounds]  # right x-axis edges (x2)
        b2_values = [b[3] for b in bounds]  # top y-axis edges (y2)

        # Compute intersection region
        ox1 = max(a1_values)
        oy1 = max(b1_values)
        ox2 = min(a2_values)
        oy2 = min(b2_values)

        # Validate positive area
        if ox1 < ox2 and oy1 < oy2:
            return Rectangle(x=ox1, y=oy1, width=ox2 - ox1, height=oy2 - oy1)
        return None

    @staticmethod
    def _compute_y_measure(
        intervals: list[tuple[Number, Number]], threshold: int = 1
    ) -> float:
        """
        Compute total vertical measure (length) covered by at least threshold intervals.

        Args:
            intervals: list of (y1, y2) tuples representing vertical spans
            threshold: minimum number of overlapping intervals to count coverage
                    - 1: total union length
                    - 2: total overlap length
        """
        if not intervals:
            return 0.0

        edges: list[tuple[Number, int]] = []
        for y1, y2 in intervals:
            edges.append((y1, +1))
            edges.append((y2, -1))

        edges.sort(key=lambda edge: (edge[0], -edge[1]))

        total = 0.0
        count = 0
        prev_y: Optional[float] = None

        for y, typ_y in edges:
            if prev_y is not None and count >= threshold and y > prev_y:
                total += y - prev_y
            count += typ_y
            prev_y = y

        return total

    def _sweep(
        self, on_step: Callable[[Number, Number, list[tuple[Number, Number]]], None]
    ) -> None:
        """
        Horizontal sweep-line function.

        Calls `on_step(x_start, x_end, active_intervals)` for each horizontal slice.

        Args:
            on_step: Function receiving:
                - x_start: left x-coordinate of slice
                - x_end: right x-coordinate of slice
                - active_intervals: list of (y1, y2) intervals active in this slice
        """
        events = []
        for rect in self.rectangles:
            x1, y1, x2, y2 = self._get_rect_bounds(rect)
            events.append((x1, y1, y2, 1))  # entering
            events.append((x2, y1, y2, -1))  # leaving

        events.sort(key=lambda e: (e[0], -e[3]))  # right edges first for open intervals

        active_intervals: list[tuple[Number, Number]] = []
        prev_x: Optional[Number] = None

        for x, y1, y2, typ in events:
            if prev_x is not None and x > prev_x:
                on_step(prev_x, x, active_intervals)

            if typ == 1:
                active_intervals.append((y1, y2))
            else:
                active_intervals.remove((y1, y2))

            prev_x = x

    def is_point_covered(self, x: Number, y: Number) -> bool:
        """
        Check if a point is covered by any rectangle.
        Returns:
            True if the point is covered by at least one rectangle, otherwise False.
        """
        for rect in self.rectangles:
            x1, y1, x2, y2 = self._get_rect_bounds(rect)
            if x1 < x < x2 and y1 < y < y2:
                return True
        return False

    def _find_multi_overlaps(
        self, group_size: Optional[int] = None
    ) -> list[tuple[int, ...]]:
        """
        Find groups of rectangles that overlap in a common area.

        Args:
            group_size:
                - If None (default), return *all* overlap groups of size ≥ 2.
                - If an integer (e.g., 2 or 3), return only groups of that exact size.

        Returns:
            A list of tuples (i, j, ...) representing rectangles that share
            an overlapping region together.
        """
        if len(self.rectangles) < 2:
            return []

        rect_bounds = [self._get_rect_bounds(r) for r in self.rectangles]
        overlaps: set[tuple[int, ...]] = set()

        def on_step(x1: float, x2: float, active_intervals: list[tuple[float, float]]):
            # Find rectangles that are active horizontally in this x-span
            active_indices = [
                i
                for i, (rx1, _, rx2, _) in enumerate(rect_bounds)
                if (min(rx2, x2) - max(rx1, x1))
                > 0  # enforce overlap 0 if rects share only an edge
            ]
            if len(active_indices) < 2:
                return

            # Build vertical enter/exit events
            y_events = []
            for idx in active_indices:
                _, y1, _, y2 = rect_bounds[idx]
                y_events.append((y1, +1, idx))
                y_events.append((y2, -1, idx))
            y_events.sort(key=lambda e: (e[0], -e[1]))

            active_y: set[int] = set()

            for y, typ_y, idx in y_events:
                if typ_y == +1:
                    active_y.add(idx)
                    if len(active_y) >= 2:
                        # Default: all sizes, otherwise only specified
                        sizes = (
                            range(2, len(active_y) + 1)
                            if group_size is None
                            else [group_size]
                        )
                        for r in sizes:
                            if r <= len(active_y):
                                for combo in combinations(sorted(active_y), r):
                                    # Verify that candidates overlap and not just touch
                                    candidates = [rect_bounds[i] for i in combo]
                                    oy1 = max(rb[1] for rb in candidates)
                                    oy2 = min(rb[3] for rb in candidates)
                                    if (oy2 - oy1) > 0:
                                        overlaps.add(combo)

                else:
                    active_y.discard(idx)

        # Perform horizontal sweep
        self._sweep(on_step)

        return sorted(overlaps)

    def find_overlaps(self) -> list[tuple[int, int]]:
        """
        Find all pairs of rectangles that overlap.

        Returns:
            A list of index pairs (i, j) with indices of overlapping rectangles.
        """
        return cast(list[tuple[int, int]], self._find_multi_overlaps(group_size=2))

    def get_overlap_regions(self) -> list[dict]:
        """
        Find actual overlap regions between rectangles.
        Returns:
            List of dicts containing:
            - 'rect_indices': tuple of rectangle indices
            - 'region': Rectangle instance, corresponding to the overlapping rectangle region.
        """

        overlapping_rectangles = self._find_multi_overlaps()

        overlap_regions = []

        for combination in overlapping_rectangles:
            rects = [self.rectangles[i] for i in combination]
            overlap = self._compute_overlap(rects)
            if overlap:
                overlap_regions.append({"rect_indices": combination, "region": overlap})
        return overlap_regions

    def find_max_overlap_point(self) -> dict[str, Optional[Number]]:
        """
        Find a point covered by maximum number of rectangles. The overlapping area is a rectangle, returns centre of this rectangle.
        Returns:
            A dict with 'x', 'y', 'count' keys. Corresponding to point coordinates and count of overlapping rectangles

        Note:
            - If multiple regions have the same maximum overlap count, the first encountered region (based on input rectangle order) is chosen deterministically.
            - If no rectangles overlap, returns {'x': None, 'y': None, 'count': None}.
        """

        overlapping_rectangles = self._find_multi_overlaps()
        if not overlapping_rectangles:
            return {"x": None, "y": None, "count": None}
        max_overlap = max(overlapping_rectangles, key=len)
        rects = [self.rectangles[i] for i in max_overlap]
        region = self._compute_overlap(rects)
        if region is None:
            return {"x": None, "y": None, "count": None}
        x = region.x + region.width / 2
        y = region.y + region.height / 2
        count = len(max_overlap)

        return {"x": x, "y": y, "count": count}

    def calculate_coverage_area(self) -> float:
        """Total area covered by at least one rectangle."""
        total_area = 0.0

        def on_step(x1: float, x2: float, intervals: list[tuple[float, float]]):
            nonlocal total_area
            dx = x2 - x1
            total_area += dx * self._compute_y_measure(
                intervals, threshold=1
            )  # compute y interval union

        self._sweep(on_step)
        return total_area

    def calculate_overlap_area(self) -> float:
        """Total area covered by two or more rectangles."""
        total_area = 0.0

        def on_step(x1: float, x2: float, intervals: list[tuple[float, float]]):
            nonlocal total_area
            dx = x2 - x1
            total_area += dx * self._compute_y_measure(
                intervals, threshold=2
            )  # compute y interval overlaps

        self._sweep(on_step)
        return total_area

    def get_stats(self) -> dict[str, Number]:
        """
        Get coverage statistics.
        Returns a dict containing:
            - 'total_rectangles': int
            - 'overlapping_pairs': int
            - 'total_area': float (union area)
            - 'overlap_area': float (sum of all overlap regions)
            - 'coverage_efficiency': float (total_area / sum_of_individual_areas)
        """
        total_rectangles = len(self.rectangles)
        overlapping_pairs = len(self.find_overlaps())
        total_area = self.calculate_coverage_area()
        overlap_area = self.calculate_overlap_area()
        sum_of_individual_areas = sum(
            self.calculate_rectangle_area(r) for r in self.rectangles
        )
        coverage_eff = (
            total_area / sum_of_individual_areas if sum_of_individual_areas > 0 else 0.0
        )

        return {
            "total_rectangles": total_rectangles,
            "overlapping_pairs": overlapping_pairs,
            "total_area": total_area,
            "overlap_area": overlap_area,
            "coverage_efficiency": coverage_eff,
        }
