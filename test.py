import unittest
import numpy as np
from solution import RouteOptimizer

class TestRouteOptimizer(unittest.TestCase):
    def setUp(self):
        self.square_coords = [(0,0), (0,1), (1,1), (1,0)]
        self.optimizer = RouteOptimizer(self.square_coords)
        
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
        route, _ = self.optimizer.find_route()
        
        self.assertEqual(route[0], route[-1])
        
        route_set = set(route[:-1])  # This excludes the last point (i.e. return to start)
        self.assertEqual(len(route_set), len(self.square_coords))
        self.assertEqual(route_set, set(range(len(self.square_coords))))
    
    def test_route_distance(self):
        route, distance = self.optimizer.find_route()
        
        manual_distance = 0
        for i in range(len(route) - 1):
            manual_distance += self.optimizer.distances[route[i]][route[i + 1]]
        
        self.assertAlmostEqual(distance, manual_distance)
    
    def test_empty_coordinates(self):
        RouteOptimizer([]) # it should not raise errors so just leaving this here for now

    def test_single_coordinate(self):
        optimizer = RouteOptimizer([(0,0)])
        route, distance = optimizer.find_route()
        self.assertEqual(route, [0, 0])
        self.assertEqual(distance, 0)

if __name__ == '__main__':
    unittest.main()