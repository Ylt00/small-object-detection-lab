from pathlib import Path
import tempfile
import unittest

from sodlab.dataset import validate_yolo_dataset
from sodlab.synthetic import SyntheticDatasetConfig, generate_dataset


class SyntheticDatasetTests(unittest.TestCase):
    def _generate(self, root: Path) -> Path:
        return generate_dataset(
            SyntheticDatasetConfig(
                width=64,
                height=64,
                train_images=3,
                val_images=2,
                test_images=2,
                seed=7,
            ),
            root,
        )

    def test_generated_dataset_is_valid(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            data_yaml = self._generate(Path(temporary_directory))
            report = validate_yolo_dataset(data_yaml)

            self.assertTrue(report.is_valid, report.errors)
            self.assertEqual(report.images, 7)
            self.assertEqual(report.labels, 7)
            self.assertGreater(report.objects, 7)
            self.assertEqual(sum(report.split_images.values()), 7)

    def test_invalid_class_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            data_yaml = self._generate(Path(temporary_directory))
            label_path = data_yaml.parent / "labels" / "train" / "train_0000.txt"
            label_path.write_text("999 0.5 0.5 0.2 0.2\n", encoding="utf-8")

            report = validate_yolo_dataset(data_yaml)

            self.assertFalse(report.is_valid)
            self.assertTrue(any("class id 999" in error for error in report.errors))


if __name__ == "__main__":
    unittest.main()
