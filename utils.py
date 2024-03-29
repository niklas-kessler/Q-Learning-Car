import numpy as np
import math


def line_intersection(line1_start, line1_end, line2_start, line2_end):
    def orientation(p, q, r):
        val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
        if val == 0:
            return 0
        elif val > 0:
            return 1
        else:
            return 2

    p1, q1 = line1_start, line1_end
    p2, q2 = line2_start, line2_end

    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)

    if o1 != o2 and o3 != o4:
        # Lines intersect
        x1, y1 = p1
        x2, y2 = q1
        x3, y3 = p2
        x4, y4 = q2

        # Calculate the intersection point
        x_intersect = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / ((x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4))
        y_intersect = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / ((x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4))

        return x_intersect, y_intersect
    else:
        # Lines do not intersect
        return 0, 0


def point_to_line_distance(v, w, p):
    """
    This function takes startpoint v and endpoint w of a line-segment and point p and returns the distance between point
    and line. Translated to python from
    https://stackoverflow.com/questions/849211/shortest-distance-between-a-point-and-a-line-segment.
    """
    v = np.asarray(v)
    w = np.asarray(w)
    p = np.asarray(p)

    l = np.dot(w - v, w - v)  # squared_length of the line

    if l == 0:  # case v == w
        return np.linalg.norm(p - v)

    t = max(0, min(1, np.dot(p-v, w-v) / l))  # clip t between 0 and 1 to handle points outside the segment vw.
    projection = v + t * (w-v)

    return np.linalg.norm(p-projection)



