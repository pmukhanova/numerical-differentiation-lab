"""Проверки точности формул и состава эксперимента."""

import math
import unittest

from main import FUNCTIONS, METHODS, calculate, steps


class DifferentiationTests(unittest.TestCase):
    def test_steps_match_assignment(self):
        hs = steps()
        self.assertEqual(len(hs), 21)
        self.assertEqual(hs[0], 1.0)
        self.assertEqual(hs[-1], 2 ** -20)
        self.assertTrue(all(a == 2 * b for a, b in zip(hs, hs[1:])))

    def test_polynomial_exactness(self):
        # Формула порядка p точна на полиномах степени <= p.
        for method, degree in zip(METHODS, (1, 1, 2, 4, 6)):
            for power in range(degree + 1):
                with self.subTest(method=method.__name__, power=power):
                    x = 0.75
                    expected = power * x ** (power - 1) if power else 0.0
                    actual = method(lambda t: t ** power, x, 0.125)
                    self.assertAlmostEqual(actual, expected, delta=1e-12)

    def test_convergence_orders_before_roundoff(self):
        # exp имеет ненулевые производные всех порядков.
        # При делении h пополам ошибка должна уменьшаться примерно в 2**p раз.
        for method, order in zip(METHODS, (1, 1, 2, 4, 6)):
            with self.subTest(method=method.__name__):
                errors = [abs(method(math.exp, 0.4, h) - math.exp(0.4))
                          for h in (0.2, 0.1)]
                measured = math.log2(errors[0] / errors[1])
                self.assertAlmostEqual(measured, order, delta=0.06)

    def test_derivatives_at_known_points(self):
        expected_at_zero = (0.0, 0.0, 0.0, 1 / 3, 1 / (2 * math.sqrt(3)))
        for function, expected in zip(FUNCTIONS, expected_at_zero):
            self.assertAlmostEqual(function.derivative(0), expected, places=14)

    def test_derivative_formulas_against_independent_difference(self):
        # Проверка цепного правила в нескольких ненулевых точках.
        for function in FUNCTIONS:
            for x in (0.4, 1.0, 1.7):
                with self.subTest(function=function.name, x=x):
                    h = 1e-5
                    approximate = (function.value(x + h) - function.value(x - h)) / (2 * h)
                    self.assertAlmostEqual(approximate, function.derivative(x), delta=1e-8)

    def test_experiment_shape_and_finite_errors(self):
        for function in FUNCTIONS:
            reference, rows = calculate(function, 1.0)
            self.assertTrue(math.isfinite(reference))
            self.assertEqual(len(rows), 21)
            for row in rows:
                self.assertEqual(len(row["values"]), 5)
                self.assertEqual(len(row["errors"]), 5)
                self.assertTrue(all(math.isfinite(e) and e >= 0 for e in row["errors"]))


if __name__ == "__main__":
    unittest.main()
