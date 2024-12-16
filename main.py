from solution import read_locations, RouteOptimizer, visualize_route, NearestNeighborStrategy, TwoOptStrategy
from datetime import datetime
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', type=str, help='Name of the file containing x, y coordinates')
    parser.add_argument('--visualize', action='store_true', help='Visualize the route')
    parser.add_argument('--nn', action='store_true', help='Use nearest neighbor as a strategy')
    parser.add_argument('--two_opt', action='store_true', help='Use 2-opt as a strategy')
    args = parser.parse_args()

    locations = read_locations(args.filename)
    if args.nn:
        optimizer = RouteOptimizer(locations, strategy=NearestNeighborStrategy())
    elif args.two_opt:
        optimizer = RouteOptimizer(locations, strategy=TwoOptStrategy(NearestNeighborStrategy()))
    else:
        optimizer = RouteOptimizer(locations)
    route, distance = optimizer.find_route()
    print(f"ROUTE\n{route}\n\nDISTANCE\n{distance:.2f}")
    if args.visualize:
        visualize_route(locations, route)