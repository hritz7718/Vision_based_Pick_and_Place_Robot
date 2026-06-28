import numpy as np

from config import (
    CUBE_CLASS_TO_LINE,
    DEFAULT_SORTING_LINE,
    PLACEMENT_DEPTH_MAX_RATIO,
    PLACEMENT_DEPTH_MIN_RATIO,
    PLACEMENT_LINE_CLEARANCE_MM,
    PLACEMENT_MARGIN_MM,
    PLACEMENT_MIN_SLOTS_PER_LINE,
    PLACEMENT_MIN_SPACING_MM,
    PLACEMENT_POINT_RADIUS_MM,
    PLACEMENT_SPACING_MM,
    ROBOT_BASE_WORLD,
    ROBOT_REACH_X_MAX_MM,
    ROBOT_REACH_X_MIN_MM,
    ROBOT_REACH_Y_MAX_MM,
    ROBOT_REACH_Y_MIN_MM,
    SORTING_LINE_COUNT,
)


class PlacementPlanner:
    def __init__(self, margin_mm=PLACEMENT_MARGIN_MM, spacing_mm=PLACEMENT_SPACING_MM):
        self.margin_mm = margin_mm
        self.spacing_mm = spacing_mm
        self.workspace_world_points = []
        self.division_line_world_points = []
        self.candidates = []
        self.candidates_by_line = {}
        self.next_index_by_line = {}
        self.next_index = 0

    def reset(self):
        self.workspace_world_points = []
        self.division_line_world_points = []
        self.candidates = []
        self.candidates_by_line = {}
        self.next_index_by_line = {}
        self.next_index = 0

    def configure(
        self,
        workspace_world_points,
        division_line_world_points,
        is_inside_workspace_func,
        is_in_placement_zone_func,
    ):
        self.workspace_world_points = list(workspace_world_points)
        self.division_line_world_points = list(division_line_world_points)
        self.candidates, self.candidates_by_line = self._build_candidate_layout(
            is_inside_workspace_func,
            is_in_placement_zone_func,
        )
        self.next_index_by_line = {
            line: 0 for line in range(1, SORTING_LINE_COUNT + 1)
        }
        self.next_index = 0

    def next_place_point(self):
        if self.next_index >= len(self.candidates):
            raise RuntimeError("No free placement point is available inside the placement zone.")

        point = self.candidates[self.next_index]
        self.next_index += 1
        return point

    def next_place_point_for_class(self, class_name):
        line, line_index, point = self.peek_place_point_for_class(class_name)
        self.commit_place_point_for_class(class_name)
        return point

    def peek_place_point_for_class(self, class_name):
        line = CUBE_CLASS_TO_LINE.get(str(class_name).lower(), DEFAULT_SORTING_LINE)
        line_candidates = self.candidates_by_line.get(line, [])
        line_index = self.next_index_by_line.get(line, 0)

        if line_index >= len(line_candidates):
            raise RuntimeError(
                f"No free placement point is available on line {line} for {class_name}."
            )

        point = line_candidates[line_index]
        return line, line_index, point

    def commit_place_point_for_class(self, class_name):
        line = CUBE_CLASS_TO_LINE.get(str(class_name).lower(), DEFAULT_SORTING_LINE)
        line_index = self.next_index_by_line.get(line, 0)
        self.next_index_by_line[line] = line_index + 1

    def reset_usage(self):
        self.next_index_by_line = {
            line: 0 for line in range(1, SORTING_LINE_COUNT + 1)
        }
        self.next_index = 0

    def get_line_for_class(self, class_name):
        return CUBE_CLASS_TO_LINE.get(str(class_name).lower(), DEFAULT_SORTING_LINE)

    def _build_candidate_layout(self, is_inside_workspace_func, is_in_placement_zone_func):
        minimum_center_spacing = PLACEMENT_POINT_RADIUS_MM * 2
        start_spacing = max(int(self.spacing_mm), int(minimum_center_spacing))
        stop_spacing = max(int(PLACEMENT_MIN_SPACING_MM), int(minimum_center_spacing))
        spacing_options = list(range(
            start_spacing,
            stop_spacing - 1,
            -2,
        ))
        if stop_spacing not in spacing_options:
            spacing_options.append(stop_spacing)

        best_candidates = []
        best_grouped = {line: [] for line in range(1, SORTING_LINE_COUNT + 1)}
        best_score = -1

        for spacing_mm in spacing_options:
            candidates = self._build_candidates(
                is_inside_workspace_func,
                is_in_placement_zone_func,
                spacing_mm,
            )
            grouped = self._group_candidates_by_line(candidates, spacing_mm)
            score = sum(
                min(len(grouped.get(line, [])), PLACEMENT_MIN_SLOTS_PER_LINE)
                for line in range(1, SORTING_LINE_COUNT + 1)
            )

            if score > best_score:
                best_score = score
                best_candidates = candidates
                best_grouped = grouped
                self.spacing_mm = spacing_mm

            if all(
                len(grouped.get(line, [])) >= PLACEMENT_MIN_SLOTS_PER_LINE
                for line in range(1, SORTING_LINE_COUNT + 1)
            ):
                self.spacing_mm = spacing_mm
                return candidates, grouped

        return best_candidates, best_grouped

    def _build_candidates(self, is_inside_workspace_func, is_in_placement_zone_func, spacing_mm):
        if len(self.workspace_world_points) < 7:
            return []

        top_left = np.array(self.workspace_world_points[0], dtype=np.float32)
        top_right = np.array(self.workspace_world_points[2], dtype=np.float32)
        bottom_left = np.array(self.workspace_world_points[6], dtype=np.float32)

        top_vector = top_right - top_left
        down_vector = bottom_left - top_left
        top_length = float(np.linalg.norm(top_vector))
        down_length = float(np.linalg.norm(down_vector))

        if top_length == 0 or down_length == 0:
            return []

        top_unit = top_vector / top_length
        down_unit = down_vector / down_length
        all_candidates = []

        row_distance = self.margin_mm
        while row_distance <= down_length - self.margin_mm:
            col_distance = self.margin_mm
            while col_distance <= top_length - self.margin_mm:
                point = top_left + top_unit * col_distance + down_unit * row_distance
                xw = float(point[0])
                yw = float(point[1])

                if is_inside_workspace_func(xw, yw) and is_in_placement_zone_func(xw, yw):
                    all_candidates.append((xw, yw, 0))

                col_distance += spacing_mm
            row_distance += spacing_mm

        return self._select_middle_reachable_candidates(all_candidates)

    def _select_middle_reachable_candidates(self, candidates):
        if not candidates:
            return []

        reachable = [point for point in candidates if self._is_reachable_by_robot(point)]
        if reachable:
            candidates = reachable

        depths = [
            self._placement_depth_from_division_line(point[0], point[1])
            for point in candidates
        ]
        positive_depths = [depth for depth in depths if depth is not None and depth > 0]
        if not positive_depths:
            return candidates

        max_depth = max(positive_depths)
        min_depth = max(PLACEMENT_LINE_CLEARANCE_MM, max_depth * PLACEMENT_DEPTH_MIN_RATIO)
        max_middle_depth = max_depth * PLACEMENT_DEPTH_MAX_RATIO
        middle_depth = (min_depth + max_middle_depth) / 2.0

        middle_candidates = [
            point for point, depth in zip(candidates, depths)
            if depth is not None and min_depth <= depth <= max_middle_depth
        ]
        if middle_candidates:
            candidates = middle_candidates

        return sorted(
            candidates,
            key=lambda point: (
                abs(self._placement_depth_from_division_line(point[0], point[1]) - middle_depth),
                abs(point[1]),
                point[0],
            ),
        )

    def _placement_depth_from_division_line(self, xw, yw):
        if len(self.division_line_world_points) != 2:
            return None

        (x1, y1), (x2, y2) = self.division_line_world_points
        line_length = float(np.hypot(x2 - x1, y2 - y1))
        if line_length == 0:
            return None

        side = (x2 - x1) * (yw - y1) - (y2 - y1) * (xw - x1)
        return max(0.0, -side / line_length)

    def _is_reachable_by_robot(self, point):
        xw, yw, _ = point
        xr = xw - float(ROBOT_BASE_WORLD[0])
        yr = yw - float(ROBOT_BASE_WORLD[1])

        return (
            ROBOT_REACH_X_MIN_MM <= xr <= ROBOT_REACH_X_MAX_MM
            and ROBOT_REACH_Y_MIN_MM <= yr <= ROBOT_REACH_Y_MAX_MM
        )

    def _group_candidates_by_line(self, candidates, spacing_mm):
        grouped = {line: [] for line in range(1, SORTING_LINE_COUNT + 1)}

        if not candidates:
            return grouped

        rows = []

        for point in sorted(candidates, key=lambda item: (item[1], item[0])):
            if not rows or abs(rows[-1][0][1] - point[1]) > spacing_mm * 0.5:
                rows.append([point])
            else:
                rows[-1].append(point)

        for row_index, row in enumerate(rows[:SORTING_LINE_COUNT], start=1):
            grouped[row_index] = row

        return grouped
