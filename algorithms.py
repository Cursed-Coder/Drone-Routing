import random as rnd
from utils import *
import sympy as sp
import math


class UAV:
    def __init__(self, id, battery_limit, weight_capacity):
        self.id = id
        self.battery_limit = battery_limit
        self.weight_capacity = weight_capacity
        self.current_battery = battery_limit
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


def find_best_candidate(route, battery_limit, weight, weight_capacity, unvisited_nodes, E):
    # print("ROUTE",route)
    # Find the best candidate node to add to the route
    best_candidate = None
    best_ratio = 0
    # print(unvisited_nodes)

    for unvisited_node in unvisited_nodes:
        for i in range(len(route) - 1):
            # print(unvisited_nodes[unvisited_node][2])
            # print("routei",route[i])
            # print("routei+1",route[i+1])
            distance_a_c = euclidean_distance(route[i], unvisited_nodes[unvisited_node][2])
            distance_c_b = euclidean_distance(unvisited_nodes[unvisited_node][2], route[i + 1])
            distance_a_b = euclidean_distance(route[i], route[i + 1])

            new_distance = distance_a_c + distance_c_b - distance_a_b
            total_energy = calculate_total_energy(route, E) + energy_consumption(new_distance, E)
            total_weight = weight + unvisited_nodes[unvisited_node][1]

            if total_energy <= battery_limit and total_weight <= weight_capacity:
                ratio = calculate_ratio(unvisited_nodes[unvisited_node][0], new_distance)
                if ratio > best_ratio:
                    best_candidate = (unvisited_node, i)
                    best_ratio = ratio

    return best_candidate


def calc_dist(route):
    # Calculate total energy consumption for a given route
    total_dist = 0
    for i in range(len(route) - 1):
        total_dist = total_dist + euclidean_distance(route[i], route[i + 1])
    return total_dist


def get_best_route_3(stop_coordinates, deliveries, battery_limit, weight_capacity, route_ranges, ratio):
    stop_keys = list(stop_coordinates.keys())
    max_reward = 0
    best_route = []
    best_route_2 = []
    total_distance = 0
    energy_per_unit_distance = 1
    total_dist = 0

    for i in range(len(stop_keys) - 1):
        start = stop_keys[i]
        end = stop_keys[i + 1]
        unvisited_nodes_copy = deliveries.copy()
        # print(unvisited_nodes_copy)
        weight = 0

        reward = 0
        coord1 = (stop_coordinates[start][1][0], stop_coordinates[start][1][1])
        coord2 = (stop_coordinates[end][1][0], stop_coordinates[end][1][1])

        route = [coord1, coord2]
        route_2 = [start, end]
        # # print(unvisited)
        # print(route)
        # route.insert(1,unvisited_nodes_copy["D1"][2])
        las_node = coord1
        dist = 0
        # print("route 0",route[0])
        # print("route 1",route[1])
        # print(euclidean_distance(route[0],route[1]))

        while unvisited_nodes_copy:
            best_candidate = find_best_candidate(route, battery_limit, weight, weight_capacity, unvisited_nodes_copy,
                                                 energy_per_unit_distance)
            if best_candidate:
                node, index = best_candidate
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

    total_dist = calc_dist(best_route)
    # print("Best Route", best_route)
    return best_route_2, max_reward, total_dist


def get_best_route(stop_coordinates, deliveries, battery_limit, weight_capacity, route_ranges, ratio):
    E = 1  # Energy usage per unit distance (adjust based on UAV efficiency)

    max_reward = 0
    best_route = []
    total_distance = 0
    stop_keys = list(stop_coordinates.keys())
    ##############################################################################################
    # Loop through all but the last stop to get s and s+1
    for i in range(len(stop_keys) - 1):
        s = stop_keys[i]  # Current stop
        s_next = stop_keys[i + 1]  # Next stop

        min_dist = ratio * (stop_coordinates[s_next][0] - stop_coordinates[s][0])
        # print("S:",s);
        # print("Sn:",s_next);

        # # Loop through each stop on the truck's predefined route
        # for s in stop_coordinates:
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

        available_deliveries = deliveries.copy()  # Copy deliveries to reset at each stop
        # print("For stop point ",s)

        while len(available_deliveries) > 0:
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

                value = delivery[0] / (distance_to_q + distance_from_q_to_cq)

                if value > mx:
                    best_del = q
                    mx = value

            # print("Bd",best_del)
            delivery = available_deliveries[best_del]
            # print("Delivery", delivery)
            distance_to_q = euclidean_distance(current_node, delivery[2])  # Distance from current node to delivery
            distance_from_q_to_cq = euclidean_distance(delivery[2], stop_coordinates[s_next][
                1])      # Distance from delivery to closest stop
            required_battery = E * (distance_to_q + distance_from_q_to_cq)
            total_poss_dist = current_total_distance + distance_to_q + distance_from_q_to_cq

            if current_battery >= required_battery and current_weight + delivery[
                1] <= weight_capacity and total_poss_dist >= min_dist:
                current_battery -= E * (euclidean_distance(current_node, available_deliveries[best_del][2]))
                # print("Current battery", current_battery)
                current_weight += available_deliveries[best_del][1]
                current_route.append(best_del)
                last_node = best_del
                # print("Next Node",best_del)
                current_reward += available_deliveries[best_del][0]
                current_total_distance += euclidean_distance(current_node, available_deliveries[best_del][2])
                current_node = available_deliveries[best_del][2]
                # print("Current Node",current_node)

            delivery = available_deliveries.pop(best_del)  # Remove selected delivery

        # If there's a valid last node, return to the closest stop
        if last_node:
            distance_from_last_to_closest = euclidean_distance(current_node, stop_coordinates[s_next][1])
            current_total_distance += distance_from_last_to_closest
            current_route.append(s_next)
            current_battery -= E * distance_from_last_to_closest

            if current_reward > max_reward:
                max_reward = current_reward

                best_route = current_route
                total_distance = current_total_distance  # Track the total distance travelled

    return best_route, max_reward, total_distance


def DRA_1(stop_coordinates, deliveries, num_drones, B, W, E):
    # uavs = [
    #     UAV(id=1, battery_limit=100, weight_capacity=20),
    #     # UAV(id=2, battery_limit=70, weight_capacity=60),
    #     # UAV(id=3, battery_limit=70, weight_capacity=55),
    # ]
    uav_s = [UAV(id=i, battery_limit=B, weight_capacity=W) for i in range(num_drones)]
    print(stop_coordinates)
    total_reward = 0
    ratio = 1

    deliveries_copy = deliveries.copy()

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
        best_route, reward, total_distance = get_best_route(stop_coordinates, deliveries_copy, uav.current_battery,
                                                            uav.weight_capacity, uav.route_ranges, ratio)

        if not best_route:
            break  # If no valid route, stop the loop

        # Update the UAV's battery based on the total distance travelled
        battery_usage = total_distance * E  # Adjust energy usage based on distance
        uav.current_battery -= battery_usage

        # Add the reward to the total reward
        total_reward += reward

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

        # print(f"Selected UAV: {uav.id}")
        # print(f"Best Route: {best_route}")
        # print(f"Reward: {reward}")
        # print(f"Total Distance: {total_distance}")
        # print(f"Remaining Battery: {uav.current_battery}")
        print(f"Total Reward: {total_reward}")

    print("Final Total Reward_1:", total_reward)


def DRA_2(stop_coordinates, deliveries, num_drones, B, W, E):
    # print(deliveries)
    # uav_s = [
    #     UAV(id=1, battery_limit=100, weight_capacity=20),
    #     # UAV(id=2, battery_limit=70, weight_capacity=60),
    #     # UAV(id=3, battery_limit=70, weight_capacity=55),
    # ]
    uav_s = [UAV(id=i, battery_limit=B, weight_capacity=W) for i in range(num_drones)]

    total_reward = 0
    ratio = 1

    deliveries_copy = deliveries.copy()

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
        best_route, reward, total_distance = get_best_route_3(stop_coordinates, deliveries_copy, uav.current_battery,
                                                              uav.weight_capacity, uav.route_ranges, ratio)

        if not best_route:
            break  # If no valid route, stop the loop

        # Update the UAV's battery based on the total distance travelled
        battery_usage = total_distance * E  # Adjust energy usage based on distance
        uav.current_battery -= battery_usage

        # Add the reward to the total reward
        total_reward += reward

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

        # print(f"Selected UAV: {uav.id}")
        # print(f"Best Route: {best_route}")
        # print(f"Reward: {reward}")
        # print(f"Total Distance: {total_distance}")
        # print(f"Remaining Battery: {uav.current_battery}")
        print(f"Total Reward: {total_reward}")

    print("Final Total Reward_2:", total_reward)
