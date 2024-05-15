import math


def select_uav_with_most_battery(uavs):
    # Sort UAVs by current battery in descending order and return the one with the most battery left
    if not uavs:
        return None

    sorted_uavs = sorted(uavs, key=lambda uav: uav.current_battery, reverse=True)
    return sorted_uavs[0]


# # Helper function to compute Euclidean distance
def euclidean_distance(coord1, coord2):
    return math.sqrt((coord1[0] - coord2[0]) ** 2 + (coord1[1] - coord2[1]) ** 2)
