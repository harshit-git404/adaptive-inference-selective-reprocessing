"""Basic tests that validate the project scaffold."""

import unittest

from config.config import get_config


class StructureTests(unittest.TestCase):
    """Validate simple scaffold assumptions."""

    def test_default_threshold_is_bounded(self) -> None:
        """The placeholder uncertainty threshold should stay in a valid range."""

        config = get_config()
        self.assertGreaterEqual(config["UNCERTAINTY_THRESHOLD"], 0.0)
        self.assertLessEqual(config["UNCERTAINTY_THRESHOLD"], 1.0)

    def test_placeholder_model_names_exist(self) -> None:
        """The scaffold should expose positive feature limits."""

        config = get_config()
        self.assertGreater(config["SMALL_MODEL_MAX_FEATURES"], 0)
        self.assertGreater(config["LARGE_MODEL_MAX_FEATURES"], 0)


if __name__ == "__main__":
    unittest.main()
