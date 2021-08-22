import unittest
import LordOfRings.generator as lrgen
import pytest
import numpy as np

class TestCore(unittest.TestCase):
    '''unittest for generator module'''

    def test_radius(self):
        '''tests if the radius is in the right range'''
        n = 128
        mu = n/2
        sigma = mu/3
        rmin, rmax = mu/4, mu
        c, r = lrgen.center_radius_generator(mu, sigma, rmin, rmax, n)
        self.assertAlmostEqual(r,
                               (rmin + rmax)/2,
                               delta = (rmax - rmin)/2)
    def test_center(self):
        '''tests if the center's coordinates are positive'''
        n = 128
        mu = n/2
        sigma = mu/3
        rmin, rmax = mu/4, mu
        c, r = lrgen.center_radius_generator(mu, sigma, rmin, rmax, n)
        tester = (c > 0).all()
        assert tester

    def test_circle(self):
        '''
        tests if the circle's points in the sparse matrix satisfy 
        the circle equation with a given radius and center.
        '''
        n = 128
        c, r = np.array([int(n/3), n/2]), n/4
        circle = lrgen.circle_generator(n, c, r)
        coord = np.argwhere(circle!=0)
        # The first element (0) is for the row (y) the second element (1) is for the row (x)
        Xcoord = coord[:, 0] + 1
        Ycoord = coord[:, 1] + 1
        tester = np.sqrt((Xcoord - c[0])**2 + (Ycoord - c[1])**2) - r
        assert (tester < 0.5).all()

    
    def test_pruned_circle(self):
        '''
        tests if some points are correctly eliminated.
        '''
        n = 128
        c, r = np.array([int(n/3), n/2]), n/4
        circle = lrgen.circle_generator(n, c, r)
        sum_circle = np.sum(circle)
        prn_circle = lrgen.rnd_circle_pruning(circle)
        assert sum_circle > np.sum(prn_circle)
        
if __name__ == '__main__':
    unittest.main()
