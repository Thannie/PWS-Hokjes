import numpy as np

def rank_and_filter_checkboxes(
    checkbox_data,
    expected_count,
    overlap_threshold=0.5,  # Step 1 overlap IoU threshold
    size_tolerance=0.2,     # For final ranking
    horizontal_tolerance=10  # For final ranking
):
    """
    1) Prune checkboxes that overlap more than `overlap_threshold`.
    2) If still more than `expected_count`, prune by x-position (closest to mean x-center).
    3) Rank remaining checkboxes by prior 'likelihood' logic (size/horizontal alignment),
       return top `expected_count` boxes.
    """

    if not checkbox_data:
        return []

    def compute_iou(boxA, boxB):
        """
        Intersection-over-Union for bounding boxes.
        box = (checkbox, y1, x1, x2, y2).
        """
        _, Ay1, Ax1, Ax2, Ay2 = boxA
        _, By1, Bx1, Bx2, By2 = boxB

        inter_x1 = max(Ax1, Bx1)
        inter_y1 = max(Ay1, By1)
        inter_x2 = min(Ax2, Bx2)
        inter_y2 = min(Ay2, By2)

        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h

        areaA = (Ax2 - Ax1) * (Ay2 - Ay1)
        areaB = (Bx2 - Bx1) * (By2 - By1)

        union_area = areaA + areaB - inter_area
        if union_area == 0:
            return 0
        return inter_area / union_area

    # Step 1: Prune by overlap
    # Sort to ensure consistent ordering (e.g., top-left to bottom-right)
    checkbox_data = sorted(checkbox_data, key=lambda x: (x[1], x[2]))
    pruned = []
    for box in checkbox_data:
        should_keep = True
        for kept in pruned:
            iou = compute_iou(box, kept)
            if iou > overlap_threshold:
                should_keep = False
                break
        if should_keep:
            pruned.append(box)

    # If we've hit the exact count after overlap pruning, return them (unranked).
    if len(pruned) == expected_count:
        return [cb[0] for cb in pruned]

    # Step 2: If still too many, prune by x-position alignment (keep those closest to mean x-center)
    if len(pruned) > expected_count:
        mean_x_center = np.mean([(x1 + x2) / 2 for _, y1, x1, x2, y2 in pruned])
        pruned.sort(key=lambda b: abs(((b[2] + b[3]) / 2) - mean_x_center))
        pruned = pruned[:expected_count]

    # If we've hit the exact count, we can return here
    if len(pruned) == expected_count:
        return [cb[0] for cb in pruned]

    # Step 3: Rank remaining boxes by your previous "likelihood" (size & horizontal alignment)
    centers_x = [(x1 + x2) / 2 for _, y1, x1, x2, y2 in pruned]
    heights = [(y2 - y1) for _, y1, x1, x2, y2 in pruned]
    widths = [(x2 - x1) for _, y1, x1, x2, y2 in pruned]

    mean_x_center = np.mean(centers_x)
    mean_h = np.mean(heights)
    mean_w = np.mean(widths)

    ranked = []
    for (chk, y1, x1, x2, y2) in pruned:
        cx = (x1 + x2) / 2
        w = x2 - x1
        h = y2 - y1

        # Horizontal alignment
        score_h = 1.0 if abs(cx - mean_x_center) <= horizontal_tolerance else 0.0

        # Size alignment
        w_ok = abs(w - mean_w) / mean_w <= size_tolerance
        h_ok = abs(h - mean_h) / mean_h <= size_tolerance
        score_size = 1.0 if (w_ok and h_ok) else 0.0

        total_score = score_h + score_size
        ranked.append((chk, total_score))

    # Sort by score DESC
    ranked.sort(key=lambda x: x[1], reverse=True)

    # Take top `expected_count`
    final_selection = [checkbox for checkbox, _ in ranked[:expected_count]]
    return final_selection
