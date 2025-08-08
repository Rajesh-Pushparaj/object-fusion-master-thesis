toal_object_count = 2770162
total_samples_count = 1078399

output_objs_per_sample = 30

# Number of objects in each class
car_count = 1913555
truck_count = 849916
pedestrian_count = 1829
motorcycle_count = 4012
bicycle_count = 850
no_object_count = (total_samples_count * output_objs_per_sample) - toal_object_count

# Invert the counts
car_weight = 1 / car_count
truck_weight = 1 / truck_count
pedestrian_weight = 1 / pedestrian_count
motorcycle_weight = 1 / motorcycle_count
bicycle_weight = 1 / bicycle_count
no_object_weight = 1 / no_object_count

# Normalize the weights
total_weight = (
    car_weight
    + truck_weight
    + pedestrian_weight
    + motorcycle_weight
    + bicycle_weight
    + no_object_weight
)
car_weight /= total_weight
truck_weight /= total_weight
pedestrian_weight /= total_weight
motorcycle_weight /= total_weight
bicycle_weight /= total_weight
no_object_weight /= total_weight

print("Class Weights (Normalized):")
print("Car:", car_weight)
print("Truck:", truck_weight)
print("Pedestrian:", pedestrian_weight)
print("Motorcycle:", motorcycle_weight)
print("Bicycle:", bicycle_weight)
print("No_object Class:", no_object_weight)
