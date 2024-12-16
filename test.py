import unittest
import numpy as np
from datetime import datetime, timedelta
from solution import (
    RouteOptimizer,
    Location,
    RouteValidationError,
    NearestNeighborStrategy,
    TwoOptStrategy
)

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
        
        route_set = set(route[:-1])  # Exclude the return to start
        self.assertEqual(len(route_set), len(self.square_locs))
        self.assertEqual(route_set, set(range(len(self.square_locs))))
    
    def test_route_distance(self):
        route, times, total_distance = self.optimizer.find_route()
        manual_distance = 0
        for i in range(len(route) - 1):
            manual_distance += self.optimizer.distances[route[i]][route[i + 1]]
        
        self.assertAlmostEqual(total_distance, manual_distance)
    
    def test_single_location(self):
        optimizer = RouteOptimizer(
            [Location(0, 0)],
            start_time=self.start_time
        )
        route, times, distance = optimizer.find_route()
        self.assertEqual(route, [0, 0])
        self.assertEqual(distance, 0)
        
    def test_time_window_constraints(self):
        # Interestingly while running this test I realized that I hadn't considered letting
        # the PE wait for the next location. This is a problem. This test will fail as it is.
        # I'm going to sleep on it as it's 2 AM right now.
        locations = [
            Location(0, 0),
            Location(
                0, 1,
                time_window_start=self.start_time,
                time_window_end=self.start_time + timedelta(hours=2)
            )
        ]
        
        optimizer = RouteOptimizer(locations, start_time=self.start_time)
        route, times, distance = optimizer.find_route()
        
        self.assertTrue(
            locations[1].time_window_start <= times[1] <= locations[1].time_window_end
        )

    def test_strategy_swap(self):
        _, _, initial_distance = self.optimizer.find_route()
        
        # we switch to just nearest neighbor without 2-opt
        self.optimizer.set_strategy(NearestNeighborStrategy())
        _, _, nn_distance = self.optimizer.find_route()
        
        # 2-opt strategy should never be worse than nearest neighbor
        self.assertLessEqual(initial_distance, nn_distance)

    def test_priority_routing(self):
        locations = [
            Location(0, 0),
            Location(0, 1, priority=2),
            Location(0, 2, priority=1)
        ]
        
        optimizer = RouteOptimizer(
            locations,
            start_time=self.start_time,
            strategy=NearestNeighborStrategy()
        )
        route, _, _ = optimizer.find_route()
        
        high_priority_index = route.index(1)
        normal_priority_index = route.index(2)
        self.assertLess(high_priority_index, normal_priority_index)

    def test_service_duration(self):
        locations = [
            Location(0, 0, service_time_minutes=30),
            Location(0, 1, service_time_minutes=60)
        ]
        
        optimizer = RouteOptimizer(locations, start_time=self.start_time)
        _, times, _ = optimizer.find_route()
        
        time_diff = times[2] - times[0]
        self.assertGreaterEqual(
            time_diff,
            timedelta(minutes=90)  # 30 + 60 minutes service time
        )

    def test_max_route_duration(self):
        locations = [
            Location(0, 0),
            Location(100, 100)
        ]
        
        optimizer = RouteOptimizer(
            locations,
            start_time=self.start_time,
            max_route_duration=timedelta(minutes=1)
        )
        
        with self.assertRaises(RouteValidationError):
            optimizer.find_route()

    def test_two_opt_improvement(self):
        locations = [
            Location(0, 0),
            Location(0, 1),
            Location(1, 0),
            Location(1, 1)
        ]
        
        optimizer = RouteOptimizer(
            locations,
            start_time=self.start_time,
            strategy=NearestNeighborStrategy()
        )
        _, _, nn_distance = optimizer.find_route()
        
        optimizer.set_strategy(TwoOptStrategy(NearestNeighborStrategy()))
        _, _, two_opt_distance = optimizer.find_route()
        
        self.assertLessEqual(two_opt_distance, nn_distance)

if __name__ == '__main__':
    unittest.main()