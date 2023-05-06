
def line_intersection(l1, l2):
    line1 = [[l1.x, l1.y], [l1.x2, l1.y2]]
    line2 = [[l2.x, l2.y], [l2.x2, l2.y2]]
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
        # lines do not intersect
        pass

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div

    return x, y

