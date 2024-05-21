import random as rnd
from utils import *
import sympy as sp
import math


class UAV:
    def __init__(self, id, battery_limit, weight_capacity, weight):
        self.id = id
        self.battery_limit = battery_limit
        self.weight_capacity = weight_capacity
        self.current_battery = battery_limit
        self.weight = weight
        self.route_ranges = []  # Keeps track of (start, end) ranges for this UAV's routes

    def __repr__(self):
        return f"UAV(id={self.id}, battery_limit={self.battery_limit}, weight_capacity={self.weight_capacity}, current_battery={self.current_battery})"


def calculate_total_energy(route, E):
    # Calculate total energy consumption for a given route
    total_energy = 0
    for i in range(len(route) - 1):
        distance = euclidean_distance(route[i], route[i + 1])
        total_energy += energy_consumption(distance, E)
    return total_energy


def energy_consumption(distance, E):
    # Calculate energy consumption based on distance

    return E * distance


def calculate_ratio(reward, distance):
    # Calculate the ratio of reward to distance
    return reward / distance if distance != 0 else float('inf')


# def find_best_candidate(route, battery_limit, weight, weight_capacity, unvisited_nodes, E):
#     # print("ROUTE",route)
#     # Find the best candidate node to add to the route
#     best_candidate = None
#     best_ratio = 0
#     # print(unvisited_nodes)
#
#     for unvisited_node in unvisited_nodes:
#         for i in range(len(route) - 1):
#             # print(unvisited_nodes[unvisited_node][2])
#             # print("routei",route[i])
#             # print("routei+1",route[i+1])
#             distance_a_c = euclidean_distance(route[i], unvisited_nodes[unvisited_node][2])
#             distance_c_b = euclidean_distance(unvisited_nodes[unvisited_node][2], route[i + 1])
#             distance_a_b = euclidean_distance(route[i], route[i + 1])
#
#             new_distance = distance_a_c + distance_c_b - distance_a_b
#             total_energy = calculate_total_energy(route, E) + energy_consumption(new_distance, E)
#             total_weight = weight + unvisited_nodes[unvisited_node][1]
#
#             if total_energy <= battery_limit and total_weight <= weight_capacity:
#                 ratio = calculate_ratio(unvisited_nodes[unvisited_node][0], new_distance)
#                 if ratio > best_ratio:
#                     best_candidate = (unvisited_node, i)
#                     best_ratio = ratio
#
#     return best_candidate

def find_best_candidate(route, route_2, route_energy, battery_limit, weight, weight_capacity, unvisited_nodes, E,
                        deliveries_2, debug):
    # Find the best candidate node to add to the route
    best_candidate = None
    best_ratio = 0
    n = len(route)
    suffix_sum = [0] * n
    best_candidate_add_energy = 0

    # suffix_sum[-1] = lst[-1]

    for i in range(n - 2, 0, -1):
        suffix_sum[i] = deliveries_2[route_2[i]][1] + suffix_sum[i + 1]
    if debug:
        print("                      ")
        print("Suffix_sum", suffix_sum)
        print("route_2", route_2)
        print("route_energy", route_energy)
    for unvisited_node in unvisited_nodes:
        for i in range(len(route) - 1):
            distance_a_c = euclidean_distance(route[i], unvisited_nodes[unvisited_node][2])
            distance_c_b = euclidean_distance(unvisited_nodes[unvisited_node][2], route[i + 1])
            distance_a_b = euclidean_distance(route[i], route[i + 1])
            if debug:
                print("distance_a_c", distance_a_c)
                print("distance_a_b", distance_a_b)
                print("distance_c_b", distance_c_b)
                print("route_i ", route[i])
                print("route_i+1 ", route[i + 1])

            new_distance = distance_a_c + distance_c_b - distance_a_b
            cumulative_distance = sum(euclidean_distance(route[j], route[j + 1]) for j in range(i))

            # Calculate the cumulative distance from start to the candidate node
            if debug:
                print("Considering for edge joining ", i, " and ", i + 1)
                print("unvisited_node", unvisited_node)
                print("cumulative distance", cumulative_distance)
                print("later weight", later_weight)
                print("weight", weight)
                print("E", E)
            # print("later_weight", later_weight)
            delivery_weight = unvisited_nodes[unvisited_node][1]
            cumulative_distance += distance_a_c
            later_weight = suffix_sum[i + 1]

            # Calculate additional energy

            additional_energy = (E * cumulative_distance * delivery_weight
                                 + new_distance * E * (weight + later_weight))

            total_energy = route_energy + additional_energy
            total_weight = weight + unvisited_nodes[unvisited_node][1]
            if debug:
                print("route_energy", route_energy)
                print("total_energy", total_energy)

            if total_energy <= battery_limit and total_weight <= weight_capacity:
                ratio = unvisited_nodes[unvisited_node][0] / additional_energy
                if ratio > best_ratio:
                    best_candidate_add_energy = additional_energy
                    best_candidate = (unvisited_node, i)
                    best_ratio = ratio
    if debug:
        print("best candidate", best_candidate)
    # route_energy += best_candidate_add_energy
    # print("route_energy_2", route_energy)
    return best_candidate, best_candidate_add_energy


def calc_dist(route):
    # Calculate total energy consumption for a given route
    total_dist = 0
    for i in range(len(route) - 1):
        total_dist = total_dist + euclidean_distance(route[i], route[i + 1])
    return total_dist


def get_best_route_3(stop_coordinates, deliveries, deliveries_2, uav, ratio, E, K, uav_stop_point_dict, debug):
    stop_keys = list(stop_coordinates.keys())
    max_reward = 0
    best_route = []
    best_route_2 = []
    best_route_energy = 0
    total_distance = 0
    # energy_per_unit_distance = 1
    total_dist = 0
    battery_limit = uav.current_battery
    weight_capacity = uav.weight_capacity
    weight_uav = uav.weight
    k = K

    print("uav_stop_point_coords", uav_stop_point_dict)

    for i in range(len(stop_keys) - 1):
        start = stop_keys[i]
        end = stop_keys[i + 1]

        if start in uav_stop_point_dict[uav]:
            continue
        unvisited_nodes_copy = deliveries.copy()

        # print(unvisited_nodes_copy)
        weight = 0

        reward = 0
        coord1 = (stop_coordinates[start][1][0], stop_coordinates[start][1][1])
        coord2 = (stop_coordinates[end][1][0], stop_coordinates[end][1][1])

        if debug:
            print("Coord", start, " ", coord1)
            print("Coord", end, " ", coord2)

        route = [coord1, coord2]
        route_2 = [start, end]
        route_energy = E * weight_uav * euclidean_distance(coord1, coord2)
        # # print(unvisited)
        # print(route)
        # route.insert(1,unvisited_nodes_copy["D1"][2])
        las_node = coord1
        dist = 0
        # print("route 0",route[0])
        # print("route 1",route[1])
        # print(euclidean_distance(route[0],route[1]))

        while unvisited_nodes_copy:
            best_candidate, best_candidate_add_energy = find_best_candidate(route, route_2, route_energy, battery_limit,
                                                                            weight_uav, weight_capacity,
                                                                            unvisited_nodes_copy,
                                                                            E, deliveries_2, debug)
            if best_candidate and k > 0:
                k = k - 1
                node, index = best_candidate
                # print("best_candidate", node)
                route_energy += best_candidate_add_energy
                route.insert(index + 1, unvisited_nodes_copy[node][2])
                route_2.insert(index + 1, node)
                # print("node",node)
                reward = reward + unvisited_nodes_copy[node][0]
                weight = weight + unvisited_nodes_copy[node][1]
                # dist = dist + euclidean_distance(las_node, unvisited_nodes_copy[node][2])
                las_node = unvisited_nodes_copy[node][2]
                unvisited_nodes_copy.pop(node)

            else:
                break

        if len(route) > 2:
            dist = dist + euclidean_distance(las_node, coord2)
            if reward > max_reward:
                max_reward = reward
                best_route = route
                best_route_2 = route_2
                best_route_energy = route_energy

    # total_dist = calc_dist(best_route)
    if debug:
        print("Best Route energy", best_route_energy)
        print("best_route_2", best_route_2)
    return best_route_2, max_reward, best_route_energy


def get_best_route(stop_coordinates, deliveries, uav, ratio, E, K, uav_stop_point_dict, debug):
    max_reward = 0
    best_route = []
    total_distance = 0
    total_energy_consumed = 0
    stop_keys = list(stop_coordinates.keys())
    k = K
    battery_limit = uav.current_battery
    weight_capacity = uav.weight_capacity
    weight_uav = uav.weight
    ##############################################################################################
    # Loop through all but the last stop to get s and s+1
    for i in range(len(stop_keys) - 1):
        s = stop_keys[i]  # Current stop
        s_next = stop_keys[i + 1]  # Next stop

        if s in uav_stop_point_dict[uav]:
            continue

        # min_dist = ratio * (stop_coordinates[s_next][0] - stop_coordinates[s][0])
        # print("S:",s);
        # print("Sn:",s_next);

        # Loop through each stop on the truck's predefined route
        # for s in stop_coordinates:///
        # if is_within_range(stop_coordinates[s][1], route_ranges):
        #     continue  # Skip stops within existing route ranges

        current_route = []
        current_reward = 0
        current_battery = battery_limit
        current_weight = 0
        current_node = stop_coordinates[s][1]
        last_node = None
        current_total_distance = 0
        current_route.append(s)
        total_weight = 0
        energy_consumed = 0

        available_deliveries = deliveries.copy()  # Copy deliveries to reset at each stop
        # print("For stop point ",s)

        while len(available_deliveries) > 0 and k > 0:
            has_constraint = False
            max_value = -float("inf")
            next_node = None

            #     # Find the best delivery (node) that meets constraints and is not within any existing range
            mx = -1e9
            best_del = None
            for q in available_deliveries:
                # if is_within_range(q["coordinates"], route_ranges):
                #         #     continue  # Skip nodes within existing route ranges

                delivery = available_deliveries[q]
                # distance_to_q = euclidean_distance(current_node, current_node)  # Distance from current node to delivery
                # print("Deliveryyyyy",q)
                distance_to_q = euclidean_distance(current_node, delivery[2])  # Distance from current node to delivery
                distance_from_q_to_cq = euclidean_distance(delivery[2], stop_coordinates[s_next][
                    1])  # Distance from delivery to closest stop
                # required_battery = E * (distance_to_q + distance_from_q_to_cq)
                total_poss_dist = current_total_distance + distance_to_q + distance_from_q_to_cq

                value = delivery[0] / ((current_total_distance + distance_to_q) * delivery[1] * E + (
                        distance_to_q + distance_from_q_to_cq) * weight_uav * E)

                if value > mx:
                    best_del = q
                    mx = value

            # print("Bd", best_del)

            delivery = available_deliveries[best_del]
            distance_to_q = euclidean_distance(current_node, delivery[2])  # Distance from current node to delivery
            distance_from_q_to_cq = euclidean_distance(delivery[2], stop_coordinates[s_next][
                1])  # Distance from delivery to closest stop
            # required_battery = E * (total_weight + ) * (distance_to_q + distance_from_q_to_cq)
            total_poss_dist = current_total_distance + distance_to_q + distance_from_q_to_cq
            weight_of_delivery = delivery[1]
            total_poss_energy = energy_consumed + (current_total_distance + distance_to_q) * (
                    weight_of_delivery + weight_uav) * E + distance_from_q_to_cq * weight_uav * E
            # print("Delivery", delivery)

            if current_battery >= total_poss_energy and current_weight + delivery[
                1] <= weight_capacity:
                energy_consumed += (
                                           current_total_distance + distance_to_q) * E * weight_of_delivery + distance_to_q * weight_uav * E
                # current_battery -= E * (euclidean_distance(current_node, available_deliveries[best_del][2]))
                # print("Current battery", current_battery)
                k = k - 1
                current_weight += available_deliveries[best_del][1]
                current_route.append(best_del)
                last_node = best_del
                # print("Next Node",best_del)
                current_reward += available_deliveries[best_del][0]
                current_total_distance += euclidean_distance(current_node, available_deliveries[best_del][2])
                current_node = available_deliveries[best_del][2]
                total_weight += available_deliveries[best_del][1]
                # print("Current Node",current_node)

            delivery = available_deliveries.pop(best_del)  # Remove selected delivery

        # If there's a valid last node, return to the closest stop
        if last_node:
            distance_from_last_to_closest = euclidean_distance(current_node, stop_coordinates[s_next][1])
            current_total_distance += distance_from_last_to_closest
            current_route.append(s_next)
            energy_consumed += distance_from_last_to_closest * weight_uav * E
            # current_battery -= E * distance_from_last_to_closest

            if current_reward > max_reward:
                max_reward = current_reward
                total_energy_consumed = energy_consumed
                best_route = current_route
                total_distance = current_total_distance  # Track the total distance travelled

    return best_route, max_reward, total_energy_consumed


def DRA_1(stop_coordinates, deliveries, num_drones, B, W, E, weight_uav, K, debug):
    # uavs = [
    #     UAV(id=1, battery_limit=100, weight_capacity=20, weight = 1),
    #     # UAV(id=2, battery_limit=70, weight_capacity=60, weight = 1),
    #     # UAV(id=3, battery_limit=70, weight_capacity=55, weight = 1),
    # ]
    uav_s = [UAV(id=i, battery_limit=B, weight_capacity=W, weight=weight_uav) for i in range(num_drones)]
    print(stop_coordinates)
    total_reward = 0
    ratio = 1

    deliveries_copy = deliveries.copy()

    uav_stop_point_dict = {}
    for u in uav_s:
        uav_stop_point_dict[u] = []

    # Main loop for selecting UAVs and determining the best route
    while True:
        # print(deliveries)
        # If there are no more available deliveries, stop
        if not deliveries_copy:
            break
        # print(deliveries)

        # Find the UAV with the most battery left
        uav = select_uav_with_most_battery(uav_s)

        # If no UAV with sufficient battery, break the loop
        if not uav or uav.current_battery <= 0:
            break

        # Get the best route with the current UAV, taking into account existing route ranges
        best_route, reward, energy_consumed = get_best_route(stop_coordinates, deliveries_copy, uav, ratio,
                                                             E, K, uav_stop_point_dict, debug)

        if not best_route:
            break  # If no valid route, stop the loop

        # Update the UAV's battery based on the total distance travelled
        battery_usage = energy_consumed  # Adjust energy usage based on distance
        uav.current_battery -= battery_usage

        # Add the reward to the total reward
        total_reward += reward
        launch_pt = best_route[0]
        uav_stop_point_dict[uav].append(launch_pt)

        # Remove deliveries in the best route from the overall deliveries set
        for node in best_route:
            if node in deliveries_copy:
                # print("Node",node)
                deliveries_copy.pop(node)
                # print(deliveries)

        # Update the UAV's route ranges with the new start and end points
        # if len(best_route) > 1:
        # start = best_route[0]
        # end = best_route[-1]
        # uav.route_ranges.append((start, ord(end[0])))  # Use ASCII to get numerical order
        if debug:
            print(f"Selected UAV: {uav.id}")
            print(f"Best Route: {best_route}")
            print(f"Reward: {reward}")
            # print(f"Total Distance: {total_distance}")
            print(f"Remaining Battery: {uav.current_battery}")
            print(f"Total Reward: {total_reward}")

    print("Final Total Reward_1:", total_reward)
    return total_reward


def DRA_2(stop_coordinates, deliveries, num_drones, B, W, E, weight_uav, K, debug):
    # print(deliveries)
    # uav_s = [
    #     UAV(id=1, battery_limit=100, weight_capacity=20),
    #     # UAV(id=2, battery_limit=70, weight_capacity=60),
    #     # UAV(id=3, battery_limit=70, weight_capacity=55),
    # ]
    uav_s = [UAV(id=i, battery_limit=B, weight_capacity=W, weight=weight_uav) for i in range(num_drones)]

    total_reward = 0
    ratio = 1
    uav_stop_point_dict = {}
    for u in uav_s:
        uav_stop_point_dict[u] = []
    deliveries_copy = deliveries.copy()
    deliveries_copy_2 = deliveries.copy()

    # Main loop for selecting UAVs and determining the best route
    while True:
        # print(deliveries)
        # If there are no more available deliveries, stop
        if not deliveries_copy:
            break
        # print(deliveries)

        # Find the UAV with the most battery left
        uav = select_uav_with_most_battery(uav_s)

        # If no UAV with sufficient battery, break the loop
        if not uav or uav.current_battery <= 0:
            break
        # def get_best_route_3(stop_coordinates, deliveries, deliveries_2, uav,
        #                      route_ranges, ratio, E, K, uav_stop_point_dict, debug):
        # Get the best route with the current UAV, taking into account existing route ranges
        # def get_best_route_3(stop_coordinates, deliveries, deliveries_2, uav, ratio, E, K, uav_stop_point_dict, debug):

        best_route, reward, energy_consumed = get_best_route_3(stop_coordinates, deliveries_copy, deliveries_copy_2,
                                                               uav, ratio, E, K, uav_stop_point_dict, debug)

        if not best_route:
            break  # If no valid route, stop the loop

        # Update the UAV's battery based on the total distance travelled
        # battery_usage = total_distance * E  # Adjust energy usage based on distance
        uav.current_battery -= energy_consumed

        # Add the reward to the total reward
        total_reward += reward
        launch_pt = best_route[0]
        uav_stop_point_dict[uav].append(launch_pt)

        # Remove deliveries in the best route from the overall deliveries set
        for node in best_route:
            if node in deliveries_copy:
                # print("Node",node)
                deliveries_copy.pop(node)
                # print(deliveries)

        # Update the UAV's route ranges with the new start and end points
        # if len(best_route) > 1:
        # start = best_route[0]
        # end = best_route[-1]
        # uav.route_ranges.append((start, ord(end[0])))  # Use ASCII to get numerical order
        if debug:
            print(f"Selected UAV: {uav.id}")
            print(f"Best Route: {best_route}")
            print(f"Reward: {reward}")
            # print(f"Total Distance: {total_distance}")
            print(f"Remaining Battery: {uav.current_battery}")
            print(f"Total Reward: {total_reward}")

    print("Final Total Reward_2:", total_reward)
    return total_reward


# import math
# import numpy as np


# class UAV:
#     def __init__(self, id, battery_limit, weight_capacity, weight):
#         self.id = id
#         self.battery_limit = battery_limit
#         self.weight_capacity = weight_capacity
#         self.current_battery = battery_limit
#         self.weight = weight
#         self.route_ranges = []  # Keeps track of (start, end) ranges for this UAV's routes
#
#     def __repr__(self):
#         return f"UAV(id={self.id}, battery_limit={self.battery_limit}, weight_capacity={self.weight_capacity}, current_battery={self.current_battery})"


# def euclidean_distance(p1, p2):
#     return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
#

def find_best_stop_pair(truck_stops, stop_keys, deliveries, uavs):
    results = []

    for delivery in deliveries:
        d_profit = deliveries[delivery][0]
        d_weight = deliveries[delivery][1]
        d_coords = deliveries[delivery][2]

        min_distance = float('inf')
        best_pair = None

        for i in range(len(stop_keys) - 1):
            s = stop_keys[i]
            s_next = stop_keys[i + 1]
            start_point = truck_stops[s][1]
            end_point = truck_stops[s_next][1]

            distance_to_delivery = euclidean_distance(start_point, d_coords)
            distance_back_to_next_stop = euclidean_distance(d_coords, end_point)
            total_distance = distance_to_delivery + distance_back_to_next_stop

            for uav in uavs:
                if uav.weight_capacity >= d_weight and uav.battery_limit >= total_distance:
                    if total_distance < min_distance:
                        min_distance = total_distance
                        best_pair = (s, s_next)

        if best_pair:
            results.append((delivery, best_pair, d_profit / min_distance))
        else:
            results.append((delivery, None, 0))  # No valid UAV found for this delivery

    return results


def build_flights(truck_stops, deliveries, uavs, E):
    # Find the best stop pair for each delivery
    stop_keys = list(truck_stops.keys())
    best_stop_pairs = find_best_stop_pair(truck_stops, stop_keys, deliveries, uavs)

    # Sort deliveries based on their launch point and ratio of profit to distance
    sorted_deliveries = sorted(best_stop_pairs,
                               key=lambda x: (truck_stops[x[1][0]][0] if x[1] else float('inf'), -x[2]))
    # print("sorted deliveries", sorted_deliveries)
    uav_stop_point_dict = {u: [] for u in uavs}
    flights = []

    for uav in uavs:
        uav_flights = []
        remaining_battery = uav.battery_limit
        current_flight = []
        current_flight_dist = 0
        current_weight = 0
        current_energy = 0
        last_delivery_coords = None
        last_stop = None
        last_delivery_to_stop_distance = 0

        for delivery, stop_pair, _ in sorted_deliveries:
            if stop_pair:
                start_stop = stop_pair[0]
                next_stop = stop_pair[1]
                d_coords = deliveries[delivery][2]
                d_weight = deliveries[delivery][1]

                if start_stop in uav_stop_point_dict[uav]:
                    continue

                if last_delivery_coords is None:
                    start_point = truck_stops[start_stop][1]
                else:
                    start_point = last_delivery_coords

                next_point = truck_stops[next_stop][1]
                distance_to_delivery = euclidean_distance(start_point, d_coords)
                distance_back_to_next_stop = euclidean_distance(d_coords, next_point)

                if len(current_flight) > 0:
                    uav_wt_append_dist = (
                                distance_to_delivery + distance_back_to_next_stop - last_delivery_to_stop_distance)
                    # energy_last_append = dist_last_append * E
                    delivery_wt_extra_dist = current_flight_dist - last_delivery_to_stop_distance + distance_to_delivery
                    energy_needed = current_energy + delivery_wt_extra_dist * d_weight * E + uav_wt_append_dist * uav.weight * E
                    if (current_weight + d_weight <= uav.weight_capacity and
                            energy_needed <= remaining_battery):
                        current_flight.append(delivery)
                        current_weight += d_weight
                        current_flight_dist += uav_wt_append_dist
                        current_energy = energy_needed
                        last_delivery_coords = d_coords
                        last_delivery_to_stop_distance = distance_back_to_next_stop
                    else:
                        uav_flights.append(current_flight)
                        uav_stop_point_dict[uav].append(start_stop)
                        current_flight = []
                        current_energy += energy_needed
                        current_weight = 0
                        current_flight_dist = 0
                        last_delivery_to_stop_distance = 0
                        energy_needed = 0
                        last_delivery_coords = None
                        start_point = truck_stops[start_stop][1]
                        distance_to_delivery = euclidean_distance(start_point, d_coords)

                if start_stop in uav_stop_point_dict[uav]:
                    continue

                if len(current_flight) == 0:
                    dist_last_append = (distance_to_delivery + distance_back_to_next_stop)
                    energy_needed = (distance_to_delivery * (d_weight + uav.weight)
                                     + distance_back_to_next_stop * uav.weight) * E
                    if (current_weight + d_weight <= uav.weight_capacity and
                            energy_needed <= remaining_battery):
                        current_flight.append(delivery)
                        current_weight += d_weight
                        current_flight_dist = dist_last_append
                        current_energy = energy_needed
                        last_delivery_coords = d_coords
                        last_delivery_to_stop_distance = distance_back_to_next_stop

        if len(current_flight) > 0:
            uav_flights.append(current_flight)
        flights.append((uav, uav_flights))

    return flights


def select_best_flight(flights, deliveries):
    max_reward = -float('inf')
    best_uav = None
    best_flight = None

    for uav, uav_flights in flights:
        # print("uav", uav)
        # print("fliht",uav_flights)
        for flight in uav_flights:
            reward = sum(deliveries[d][0] for d in flight)
            if reward > max_reward:
                max_reward = reward
                best_uav = uav
                best_flight = flight

    return best_uav, best_flight, max_reward


def iterative_build_flights(truck_stops, deliveries, uavs, E):
    all_flights = []
    remaining_deliveries = deliveries.copy()
    remaining_uavs = uavs.copy()

    while remaining_uavs and remaining_deliveries:
        flights = build_flights(truck_stops, remaining_deliveries, remaining_uavs, E)
        best_uav, best_flight, max_reward = select_best_flight(flights, remaining_deliveries)

        if best_flight:
            all_flights.append((best_uav, best_flight))
            remaining_uavs.remove(best_uav)
            for delivery in best_flight:
                del remaining_deliveries[delivery]
        else:
            break

    return all_flights


# Example usage:

# Truck stop points (in order)
def DRA_3(stop_coordinates, deliveries, num_drones, B, W, E, weight_uav, K, debug):
    # truck_stops = {
    #     's1': (48.475643284870344, np.array([38.55993996, 65.74388305])),
    #     's2': (116.29400534058423, np.array([37.9622159 , 63.64426632])),
    #     's3': (203.14362653751732, np.array([70.98484424, 19.35354328])),
    #     's4': (270.8258997146956, np.array([14.73032554, 30.70628411]))
    # }
    #
    # UAVs with weight carrying capacity and battery limit (maximum distance)
    uav_s = [UAV(id=i, battery_limit=B, weight_capacity=W, weight=weight_uav) for i in range(num_drones)]

    # Deliveries with weight, profit, and coordinates
    # deliveries = {
    #     'p1': {0: 10, 1: 3.427099253612857, 2: (8.969854822794765, 96.92142909395326)},
    #     'p2': {0: 1, 1: 2.9358298593710237, 2: (94.91501069149506, 59.11756729080847)},
    #     'p3': {0: 1, 1: 2.7882057370960727, 2: (6.544310293541889, 58.08071895810054)},
    #     'p4': {0: 1, 1: 2.3161914908695618, 2: (63.20901569381385, 26.23390707996971)}
    # }

    # Build flights for UAVs
    all_flights = iterative_build_flights(stop_coordinates, deliveries, uav_s, E)
    total_reward = 0
    for uav, flight in all_flights:
        print(f"UAV {uav.id} with flight: {flight}")
        for p in flight:
            total_reward += deliveries[p][0]
    return  total_reward

