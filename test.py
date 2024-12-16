import unittest
import numpy as np
from decimal import Decimal
from solution import (
    RouteOptimizer,
    Location,
    NearestNeighborStrategy,
    TwoOptStrategy
)

class TestRouteOptimizer(unittest.TestCase):
    def setUp(self):
        """
        Create a simple square of locations for testing.
        """
        self.square_locs = [
            Location(0, 0),
            Location(0, 1),
            Location(1, 1),
            Location(1, 0)
        ]
        self.optimizer = RouteOptimizer(self.square_locs)
        
    def test_distance_matrix(self):
        """
        Test if the distance matrix is calculated correctly.
        """
        expected = np.array([
            [0, 1, np.sqrt(2), 1],
            [1, 0, 1, np.sqrt(2)],
            [np.sqrt(2), 1, 0, 1],
            [1, np.sqrt(2), 1, 0]
        ])
        
        np.testing.assert_array_almost_equal(
            self.optimizer.distances,
            expected
        )
    
    def test_route_completion(self):
        """
        Test if the route is complete, i.e.
        - It starts and ends at the same point
        - It visits all locations exactly once
        """
        route, distance = self.optimizer.find_route()
        
        
        self.assertEqual(route[0], route[-1])
        
        route_set = set(route[:-1])  # this is to exclude the start from the "have all locations been visited" check
        self.assertEqual(len(route_set), len(self.square_locs))
        self.assertEqual(route_set, set(range(len(self.square_locs))))
    
    def test_route_distance(self):
        """
        Test if the reported total distance matches manual calculation.
        """
        route, total_distance = self.optimizer.find_route()
        
        manual_distance = Decimal('0')
        for i in range(len(route) - 1):
            manual_distance += Decimal(str(self.optimizer.distances[route[i]][route[i + 1]]))
        
        self.assertAlmostEqual(float(total_distance), float(manual_distance))
    
    def test_single_location(self):
        """
        Test handling of a single location case.
        Should return a simple out-and-back route with zero distance.
        """
        optimizer = RouteOptimizer([Location(0, 0)])
        route, distance = optimizer.find_route()
        self.assertEqual(route, [0, 0])
        self.assertEqual(distance, 0)

    def test_strategy_swap(self):
        """
        Test if swapping strategies works correctly.
        2-opt should never be worse than nearest neighbor alone.
        """
        _, initial_distance = self.optimizer.find_route()
        
        # we switch to just nearest neighbor without 2-opt
        self.optimizer.set_strategy(NearestNeighborStrategy())
        _, nn_distance = self.optimizer.find_route()
        
        # 2-opt strategy should never be worse than nearest neighbor
        self.assertLessEqual(initial_distance, nn_distance)

    def test_two_opt_improvement(self):
        """
        Test if 2-opt can improve on a sub-optimal route.
        """
        locations = [
            Location(0, 0),    # Start
            Location(0, 2),    # North
            Location(2, 0),    # East
            Location(3, 3)     # Northeast but farther so NN won't take it
        ]
        
        optimizer = RouteOptimizer(
            locations,
            strategy=NearestNeighborStrategy()
        )
        _, nn_distance = optimizer.find_route()
        
        optimizer.set_strategy(TwoOptStrategy(NearestNeighborStrategy()))
        _, two_opt_distance = optimizer.find_route()
        
        # 2-opt should find a better route here
        self.assertLess(two_opt_distance, nn_distance)

    def test_empty_locations(self):
        """
        Test that an empty location list is properly rejected.
        """
        with self.assertRaises(ValueError):
            RouteOptimizer([])

if __name__ == '__main__':
    unittest.main()