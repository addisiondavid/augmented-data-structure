# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from scipy.spatial import KDTree
import time
from time import perf_counter
import random
import os
#import generated_poi_data
import csv
#from sklearn.neighbors import KDTree


def find_nearest_hospital_with_capacity(hospitals, query_point, capacity_filter):
    """
    Use scipy.spatial.KDTree to find the nearest hospital with at least the given capacity.
    Args:
        hospitals: List of tuples (latitude, longitude, name, capacity).
        query_point: (latitude, longitude) of the query point.
        capacity_filter: Minimum capacity required for a hospital.
    Returns:
        Tuple: (nearest_name, nearest_point, distance) of the nearest matching hospital.
    """
    # Pre-filter hospitals based on the capacity filter
    filtered_hospitals = [
        (point, name) for point, name, capacity in hospitals if capacity >= capacity_filter
    ]
    if not filtered_hospitals:
        return None, None, float("inf")  # No matching hospital

    # Extract points and names for the filtered hospitals
    points = [point for point, name in filtered_hospitals]
    names = [name for point, name in filtered_hospitals]

    # Build the KDTree
    kd_tree = KDTree(points)

    # Query the KDTree for the nearest point
    distance, index = kd_tree.query(query_point)

    # Retrieve the nearest hospital's name and location
    nearest_name = names[index]
    nearest_point = points[index]

    return nearest_name, nearest_point, distance

class AugmentedKDTreeNode:
    def __init__(self, point, name, capacity, left=None, right=None):
        self.point = point        # Spatial data (latitude, longitude)
        self.name = name          # Metadata: Hospital name
        self.capacity = capacity  # Metadata: Hospital capacity
        self.left = left          # Left subtree
        self.right = right        # Right subtree


def build_augmented_kd_tree(hospitals, depth=0):
    """
    Build an augmented k-d tree from the given hospital data.
    Args:
        hospitals: List of tuples (latitude, longitude, name, capacity).
        depth: Current depth in the tree for alternating splitting axis.
    Returns:
        AugmentedKDTreeNode: Root of the augmented k-d tree.
    """
    if not hospitals:
        return None

    k = 2  # Fixed dimensions (latitude, longitude)
    axis = depth % k

    # Sort by the current axis (latitude or longitude)
    hospitals.sort(key=lambda x: x[0][axis])
    median_idx = len(hospitals) // 2

    # Create a node for the median point
    median_point = hospitals[median_idx][0]
    median_name = hospitals[median_idx][1]
    median_capacity = hospitals[median_idx][2]

    return AugmentedKDTreeNode(
        point=median_point,
        name=median_name,
        capacity=median_capacity,
        left=build_augmented_kd_tree(hospitals[:median_idx], depth + 1),
        right=build_augmented_kd_tree(hospitals[median_idx + 1:], depth + 1),
    )

def generate_random_point_in_california():
    # California latitude and longitude bounds
    min_lat, max_lat = 32.5, 42.0
    min_lon, max_lon = -124.4, -114.1

    # Generate random latitude and longitude within bounds
    random_lat = random.uniform(min_lat, max_lat)
    random_lon = random.uniform(min_lon, max_lon)

    return (random_lat, random_lon)





def query_augmented_kd_tree(node, query_point, capacity_filter, depth=0):
    """
    Query the augmented k-d tree for the nearest neighbor that satisfies the capacity filter.
    Args:
        node: Current node in the k-d tree.
        query_point: (latitude, longitude) of the query point.
        capacity_filter: Minimum capacity required for a hospital.
        depth: Current depth in the tree for alternating splitting axis.
    Returns:
        Tuple: (nearest_name, nearest_point, distance) of the nearest matching hospital.
    """
    if node is None:
        return None, None, float("inf")

    # Compute squared Euclidean distance
    distance_sq = sum((qp - np) ** 2 for qp, np in zip(query_point, node.point))

    # Initialize best match
    best_name, best_point, best_distance_sq = None, None, float("inf")
    if node.capacity >= capacity_filter:
        best_name, best_point, best_distance_sq = node.name, node.point, distance_sq

    # Determine splitting axis
    axis = depth % 2

    # Traverse the next branch based on splitting axis
    next_branch = node.left if query_point[axis] < node.point[axis] else node.right
    other_branch = node.right if next_branch == node.left else node.left
    candidate_name, candidate_point, candidate_distance_sq = query_augmented_kd_tree(
        next_branch, query_point, capacity_filter, depth + 1
    )

    # Update best match
    if candidate_distance_sq < best_distance_sq:
        best_name, best_point, best_distance_sq = candidate_name, candidate_point, candidate_distance_sq

    # Check the other branch if necessary
    if (query_point[axis] - node.point[axis]) ** 2 < best_distance_sq:
        alt_name, alt_point, alt_distance_sq = query_augmented_kd_tree(
            other_branch, query_point, capacity_filter, depth + 1
        )
        if alt_distance_sq < best_distance_sq:
            best_name, best_point, best_distance_sq = alt_name, alt_point, alt_distance_sq

    return best_name, best_point, best_distance_sq ** 0.5


def generate_california_hospital_data(num_hospitals):
    """
    Generate a dataset of hospitals within California boundaries.
    Args:
        num_hospitals (int): Number of hospitals to generate.
    Returns:
        List of tuples in the format:
            ((latitude, longitude), name, capacity)
    """
    california_bounds = {
        "latitude": (32.5, 42.0),  # Latitude range
        "longitude": (-124.4, -114.1)  # Longitude range
    }

    names = [
        "Central Hospital", "West Side Hospital", "East Clinic",
        "North General", "South Memorial", "Golden State Hospital",
        "Seaside Medical", "Sierra Clinic", "Valley Health", "Bay Area Hospital",
    ]

    hospitals = []
    for name in range(num_hospitals):
        latitude = round(random.uniform(*california_bounds["latitude"]), 6)
        longitude = round(random.uniform(*california_bounds["longitude"]), 6)
        capacity = random.randint(50, 500)  # Random capacity between 50 and 500
        ran_name = random.choice(names)
        hospitals.append(((latitude, longitude), ran_name, capacity))

    return hospitals



hospitals = generate_california_hospital_data(500000)

print(hospitals)
csv_file = "kd_tree_comparison_results.csv"
augmented_runtimes = []
standard_runtimes = []
augmented_results = []
standard_results = []

for i in range(100):
    query_point = generate_random_point_in_california()
    # Build the augmented k-d tree
    tree = build_augmented_kd_tree(hospitals)
    capacity_filter = 200

    # Query the KD tree
    kd_start_time = perf_counter()
    nearest_name, nearest_point, distance = find_nearest_hospital_with_capacity(hospitals, query_point, capacity_filter)
    kd_end_time = perf_counter()

    kd_runtime = kd_end_time - kd_start_time
    standard_runtimes.append(kd_runtime)
    standard_results.append((nearest_point, distance))

    # Query the Augmented KD tree
    start_time = perf_counter()
    nearest_name, nearest_point, distance = query_augmented_kd_tree(tree, query_point, capacity_filter)
    end_time = perf_counter()

    augmented_runtime = end_time - start_time
    augmented_runtimes.append(augmented_runtime)
    augmented_results.append((nearest_point, distance))
    # Print the result
    print(f"Nearest hospital with at least {capacity_filter} beds:")
    print(f"Name: {nearest_name}, Location: {nearest_point}, Distance: {distance:.2f}")


with open(csv_file, mode="w", newline="") as file:
    writer = csv.writer(file)

    # Write header
    writer.writerow([
        "Run",
        "Augmented k-d Tree Query Time (s)", "Augmented k-d Tree Result", "Augmented k-d Tree Distance",
        "Standard k-d Tree Query Time (s)", "Standard k-d Tree Result", "Standard k-d Tree Distance"
    ])

    # Write data
    for idx in range(100):
        writer.writerow([
            idx + 1,
            augmented_runtimes[idx],
            augmented_results[idx][0],
            augmented_results[idx][1],
            standard_runtimes[idx],
            standard_results[idx][0],
            standard_results[idx][1]
        ])

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    #print_hi('PyCharm')
    print('')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
