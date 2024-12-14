import unittest
import numpy as np
from datetime import datetime, timedelta
from solution import RouteOptimizer, Location, RouteValidationError

class TestRouteOptimizer(unittest.TestCase):
    def setUp(self):
        self.square_locs = [
            Location(0, 0),
            Location(0, 1),
            Location(1, 1),
            Location(1, 0)
        ]
        self.start_time = datetime(2024, 12, 14, 8, 0)
        self.optimizer = RouteOptimizer(
            self.square_locs,
            start_time=self.start_time
        )
        
    def test_distance_matrix(self):
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
        route, times, distance = self.optimizer.find_route()
        
        self.assertEqual(route[0], route[-1])
        
        route_set = set(route[:-1])  # exclude the return to start
        self.assertEqual(len(route_set), len(self.square_locs))
        self.assertEqual(route_set, set(range(len(self.square_locs))))
    
    def test_route_distance(self):
        route, times, total_distance = self.optimizer.find_route()

        manual_distance = 0
        for i in range(len(route) - 1):
            manual_distance += self.optimizer.distances[route[i]][route[i + 1]]
        
        self.assertAlmostEqual(total_distance, manual_distance)
    
    def test_empty_locations(self):
        with self.assertRaises(ValueError):
            RouteOptimizer([])

    def test_single_location(self):
        optimizer = RouteOptimizer([Location(0, 0)], start_time=self.start_time)
        route, times, distance = optimizer.find_route()
        self.assertEqual(route, [0, 0])
        self.assertEqual(distance, 0)
        
    def test_time_window_constraints(self):
        locations = [
            Location(0, 0),
            Location(
                0, 1,
                time_window_start=self.start_time + timedelta(hours=1),
                time_window_end=self.start_time + timedelta(hours=2)
            )
        ]
        
        optimizer = RouteOptimizer(locations, start_time=self.start_time)
        route, times, distance = optimizer.find_route()
        
        self.assertTrue(
            locations[1].time_window_start <= times[1] <= locations[1].time_window_end
        )

    def test_invalid_time_windows(self):
        locations = [
            Location(0, 0),
            Location(
                0, 1,
                time_window_start=self.start_time,
                time_window_end=self.start_time
            )
        ]
        
        optimizer = RouteOptimizer(locations, start_time=self.start_time)
        with self.assertRaises(RouteValidationError):
            optimizer.find_route()

if __name__ == '__main__':
    unittest.main()