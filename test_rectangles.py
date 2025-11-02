import time

import pytest

from rectangle_analyzer import Rectangle, RectangleAnalyzer


@pytest.fixture
def single_rectangle():
    return [{"x": 0, "y": 0, "width": 2, "height": 4}]


@pytest.fixture
def overlapping_rects():
    return [
        {"x": 0, "y": 0, "width": 3, "height": 3},
        {"x": 1, "y": 1, "width": 3, "height": 3},
        {"x": 0.5, "y": 1.5, "width": 3, "height": 3},
    ]


@pytest.fixture
def disjointed_rects():
    return [
        {"x": 0, "y": 0, "width": 2, "height": 2},
        {"x": 3, "y": 0, "width": 2, "height": 2.0},
        {"x": 0, "y": 3, "width": 2, "height": 2},
    ]


@pytest.fixture
def common_side_vertical_rects():
    return [
        {"x": 0, "y": 0, "width": 2, "height": 2},
        {"x": 2, "y": 0, "width": 2, "height": 2},
    ]


@pytest.fixture
def common_side_horizontal_rects():
    return [
        {"x": 0, "y": 0, "width": 2, "height": 2},
        {"x": 0, "y": 2, "width": 2, "height": 2},
    ]


@pytest.fixture
def two_separate_overlap_rects():
    return [
        {"x": 0, "y": 0, "width": 3, "height": 3},
        {"x": 1, "y": 1, "width": 3, "height": 3},
        {"x": 5, "y": 0, "width": 3, "height": 3},
    ]


@pytest.fixture
def generate_rects():
    """Fixture to generate horizontally aligned overlapping rectangles."""

    def _generate(n=100, width=5, height=3, overlap=2):
        rects = []
        for i in range(n):
            rects.append(
                {"x": i * (width - overlap), "y": 0, "width": width, "height": height}
            )
        return rects

    return _generate


def test_rectangle_equality():
    r1 = Rectangle(1, 2, 3, 4)
    r2 = Rectangle(1.0, 2.0, 3.0, 4.0)
    r3 = Rectangle(1, 2, 3, 5)
    assert r1 == r2
    assert r1 != r3


@pytest.mark.parametrize(
    "key,bad_value,expected_exception,expected_message",
    [
        # Width
        ("width", -1, ValueError, "width and height must be > 0"),
        ("width", 0, ValueError, "width and height must be > 0"),
        ("width", "a", TypeError, "must be an int or float"),
        ("width", None, TypeError, "must be an int or float"),
        # Height
        ("height", -1, ValueError, "width and height must be > 0"),
        ("height", 0, ValueError, "width and height must be > 0"),
        ("height", None, TypeError, "must be an int or float"),
        # Coordinates
        ("x", "a", TypeError, "must be an int or float"),
        ("x", None, TypeError, "must be an int or float"),
        ("y", "b", TypeError, "must be an int or float"),
        ("y", object(), TypeError, "must be an int or float"),
    ],
)
def test_invalid_rectangle_values(key, bad_value, expected_exception, expected_message):
    """
    Test that RectangleAnalyzer (via Rectangle) raises the correct exceptions
    for invalid numeric or type inputs.
    """
    rects = [
        {"x": 0, "y": 0, "width": 2, "height": 2},
        {"x": 1, "y": 1, "width": 3, "height": 3, key: bad_value},
    ]

    with pytest.raises(expected_exception, match=expected_message):
        RectangleAnalyzer(rects)


def test_find_overlaps_normal(overlapping_rects):
    """Test that overlapping rectangles are correctly identified."""
    r = RectangleAnalyzer(overlapping_rects)

    assert sorted(r.find_overlaps()) == [(0, 1), (0, 2), (1, 2)]


@pytest.mark.parametrize(
    "fixture_name",
    [
        "common_side_horizontal_rects",
        "common_side_vertical_rects",
        "disjointed_rects",
        "single_rectangle",
    ],
)
def test_find_overlaps_edge_and_disj(request, fixture_name):
    """Test that rectangles that share only an edge or are disjointed do not overlap."""
    rects = request.getfixturevalue(fixture_name)
    r = RectangleAnalyzer(rects)
    assert r.find_overlaps() == []


def test_get_overlap_regions_normal(overlapping_rects):
    """Test that overlapping areas are identified correctly."""
    r = RectangleAnalyzer(overlapping_rects)
    regions = r.get_overlap_regions()

    expected = [
        {"rect_indices": (0, 1), "region": Rectangle(1, 1, 2, 2)},
        {"rect_indices": (0, 2), "region": Rectangle(0.5, 1.5, 2.5, 1.5)},
        {"rect_indices": (1, 2), "region": Rectangle(1, 1.5, 2.5, 2.5)},
        {"rect_indices": (0, 1, 2), "region": Rectangle(1, 1.5, 2, 1.5)},
    ]

    assert sorted(regions, key=lambda r: r["rect_indices"]) == sorted(
        expected, key=lambda r: r["rect_indices"]
    )


@pytest.mark.parametrize(
    "fixture_name",
    [
        "common_side_horizontal_rects",
        "common_side_vertical_rects",
        "disjointed_rects",
        "single_rectangle",
    ],
)
def test_get_overlap_regions_edge_and_disj(request, fixture_name):
    """Test that rectangles that share only an edge or are disjointed do not yield any overlaps."""
    rects = request.getfixturevalue(fixture_name)
    r = RectangleAnalyzer(rects)
    assert r.get_overlap_regions() == []


def test_calculate_overlap_area_overlapping_rects(overlapping_rects):
    """Test that overlap area is calculated correctly."""
    r = RectangleAnalyzer(overlapping_rects)
    assert r.calculate_overlap_area() == 8.0


@pytest.mark.parametrize(
    "fixture_name",
    [
        "common_side_horizontal_rects",
        "common_side_vertical_rects",
        "disjointed_rects",
        "single_rectangle",
    ],
)
def test_calculate_overlap_area_edge_and_disj(request, fixture_name):
    """Test that rectangles that share only an edge or are disjointed do not yield any overlaps."""
    rects = request.getfixturevalue(fixture_name)
    r = RectangleAnalyzer(rects)
    assert r.calculate_overlap_area() == 0.0


def test_calculate_coverage_area_overlapping_rects(overlapping_rects):
    """Test that coverage area is calculated correctly."""
    r = RectangleAnalyzer(overlapping_rects)
    assert r.calculate_coverage_area() == 16.0


@pytest.mark.parametrize(
    "fixture_name",
    [
        "common_side_horizontal_rects",
        "common_side_vertical_rects",
        "single_rectangle",
    ],
)
def test_calculate_coverage_area_edge_and_disj(request, fixture_name):
    """Test that coverage area is calculated correctly."""
    rects = request.getfixturevalue(fixture_name)
    r = RectangleAnalyzer(rects)
    assert r.calculate_coverage_area() == 8.0


@pytest.mark.parametrize(
    "point,expected,description",
    [
        ((-1, -1), False, "point outside all rectangles"),
        ((1.5, 2), True, "point in the overlap of all rectangles"),
        ((0.5, 0.5), True, "point inside only one rectangle"),
        ((1, 0), False, "point on bottom horizontal edge"),
        ((0, 1), False, "point on left vertical edge"),
        ((3, 0.5), False, "point on right vertical edge"),
        ((1, 4.5), False, "point on top horizontal edge"),
    ],
)
def test_is_point_covered(overlapping_rects, point, expected, description):
    """
    Test point coverage scenarios including:
    - Outside all rectangles
    - Inside overlap of all rectangles
    - Inside a single rectangle
    - On all rectangle edges
    """
    r = RectangleAnalyzer(overlapping_rects)
    assert r.is_point_covered(*point) is expected, f"Failed: {description}"


def test_find_max_overlap_point_overlapping(overlapping_rects):
    """
    Test the maximum overlap point for rectangles that do overlap.
    Should return center of the region covered by the most rectangles.
    """
    r = RectangleAnalyzer(overlapping_rects)
    result = r.find_max_overlap_point()

    expected = {"x": 2.0, "y": 2.25, "count": 3}

    assert result == expected


@pytest.mark.parametrize(
    "fixture_name",
    [
        "disjointed_rects",
        "common_side_horizontal_rects",
        "common_side_vertical_rects",
        "single_rectangle",
    ],
)
def test_find_max_overlap_point_no_overlap(request, fixture_name):
    """
    Test rectangles that are disjoint or share only a side.
    Should return None for x, y, and count since no overlaps exist.
    """
    rects = request.getfixturevalue(fixture_name)
    r = RectangleAnalyzer(rects)
    result = r.find_max_overlap_point()
    assert result == {"x": None, "y": None, "count": None}


def test_find_max_overlap_point_two_separate_overlaps(two_separate_overlap_rects):
    """
    Test scenario where three rectangles create two separate overlapping regions.
    The function should return the center of the first maximum-overlap region (overlap of 2 rectangles).
    """
    r = RectangleAnalyzer(two_separate_overlap_rects)
    result = r.find_max_overlap_point()

    expected = {"x": 2, "y": 2, "count": 2}

    assert result == expected


@pytest.mark.parametrize("n", [10, 100, 1000])
def test_stress_timing_coverage(generate_rects, n):
    """Stress test timing for calculate_coverage_area()."""
    rects = generate_rects(n=n, overlap=2)
    r = RectangleAnalyzer(rects)

    start = time.perf_counter()
    coverage = r.calculate_coverage_area()
    elapsed = time.perf_counter() - start

    print(
        f"\n[Coverage] Rectangles: {n:5d} | Area: {coverage:10.2f} | Time: {elapsed:8.5f} s"
    )

    assert coverage > 0


@pytest.mark.parametrize("n", [10, 100, 1000])
def test_stress_timing_overlap(generate_rects, n):
    """Stress test timing for calculate_overlap_area()."""
    rects = generate_rects(n=n, overlap=2)
    r = RectangleAnalyzer(rects)

    start = time.perf_counter()
    overlap = r.calculate_overlap_area()
    elapsed = time.perf_counter() - start

    print(
        f"\n[Overlap]  Rectangles: {n:5d} | Area: {overlap:10.2f} | Time: {elapsed:8.5f} s"
    )

    assert overlap >= 0


@pytest.mark.parametrize("n", [10, 100, 1000])
def test_stress_timing_multi_overlaps(generate_rects, n):
    """Stress test timing for find_multi_overlaps()."""
    rects = generate_rects(n=n, overlap=2)
    r = RectangleAnalyzer(rects)

    start = time.perf_counter()
    overlaps = r._find_multi_overlaps()
    elapsed = time.perf_counter() - start

    print(
        f"\n[MultiOverlaps] Rectangles: {n:5d} | Groups found: {len(overlaps):5d} | Time: {elapsed:8.5f} s"
    )

    assert isinstance(overlaps, list)
