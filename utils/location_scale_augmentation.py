import numpy as np
import random
from math import comb


class LocationScaleAugmentation(object):
    def __init__(self, vrange=(0., 1.), background_threshold=0.01, nPoints=4, nTimes=100000,
                 min_target_pixels=2, sigma1=0.1, sigma2=0.5):
        if nPoints != 4 or nTimes < 2 or min_target_pixels < 1:
            raise ValueError('Use a cubic curve, at least 2 samples, and a positive pixel minimum')
        self.nPoints = nPoints
        self.nTimes = nTimes
        self.vrange = vrange
        self.background_threshold = background_threshold
        self.min_target_pixels = min_target_pixels
        self.sigma1, self.sigma2 = sigma1, sigma2
        self._get_polynomial_array()

    def _get_polynomial_array(self):
        def bernstein_poly(i, n, t):
            return comb(n, i) * (t ** i) * (1 - t) ** (n - i)

        t = np.linspace(0.0, 1.0, self.nTimes)
        self.polynomial_array = np.array(
            [bernstein_poly(i, self.nPoints - 1, t) for i in range(0, self.nPoints)]).astype(np.float32)

    def get_bezier_curve(self, points):
        xPoints = np.array([p[0] for p in points])
        yPoints = np.array([p[1] for p in points])
        xvals = np.dot(xPoints, self.polynomial_array)
        yvals = np.dot(yPoints, self.polynomial_array)
        return xvals, yvals

    def non_linear_transformation_target(self, inputs, target, inverse=False, inverse_prop=0.5):
        start_point, end_point = target.min(), target.max()
        xPoints = [start_point, end_point]
        yPoints = [start_point, end_point]
        for _ in range(self.nPoints - 2):
            xPoints.insert(1, random.uniform(xPoints[0], xPoints[-1]))
            yPoints.insert(1, random.uniform(yPoints[0], yPoints[-1]))
        xvals, yvals = self.get_bezier_curve([[x, y] for x, y in zip(xPoints, yPoints)])
        if inverse and random.random() <= inverse_prop:
            xvals = np.sort(xvals)
        else:
            xvals, yvals = np.sort(xvals), np.sort(yvals)
        return np.interp(inputs, xvals, yvals)

    def location_scale_transformation(self, inputs, slide_limit=20):
        scale = np.array(max(min(random.gauss(1, self.sigma1), 1.1), 0.9), dtype=np.float32)
        location = np.array(random.gauss(0, self.sigma2), dtype=np.float32)
        location = np.clip(location, self.vrange[0] - np.percentile(inputs, slide_limit),
                           self.vrange[1] - np.percentile(inputs, 100 - slide_limit))
        return np.clip(inputs * scale + location, self.vrange[0], self.vrange[1])

    def Global_Location_Scale_Augmentation_target(self, image, target):
        image = self.non_linear_transformation_target(image, target, inverse=False)
        image = self.location_scale_transformation(image).astype(np.float32)
        return image

    def Local_Location_Scale_Augmentation_target(self, image, target, image_mask, target_mask):
        # Missing or unreliable target classes retain their original source values.
        output_image = image.copy()

        image_mask = image_mask.astype(np.int32)

        for c in np.unique(image_mask):
            if c < 0:
                continue
            source_region = image_mask == c
            target_values = target[target_mask == c]
            if target_values.size < self.min_target_pixels or np.ptp(target_values) <= 1e-8:
                continue
            output_image[source_region] = self.non_linear_transformation_target(
                image[source_region], target_values, inverse=False)

        return output_image

