from pathlib import Path
import tempfile
import unittest
import zipfile

from sodlab.data import extract_zip_safely, write_coco8_yaml, verify_sha256
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


class DatasetPreparationTests(unittest.TestCase):
    def test_sha256_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "payload.bin"
            path.write_bytes(b"content")

            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                verify_sha256(path, "0" * 64)

    def test_zip_with_parent_traversal_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            archive = Path(temporary_directory) / "unsafe.zip"
            with zipfile.ZipFile(archive, "w") as output:
                output.writestr("../escape.txt", "no")

            with self.assertRaisesRegex(ValueError, "escapes destination"):
                extract_zip_safely(archive, Path(temporary_directory) / "output")

    def test_coco8_yaml_is_written_for_valid_layout(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "coco8"
            for relative in ("images/train", "images/val", "labels/train", "labels/val"):
                (root / relative).mkdir(parents=True)
            data_yaml = write_coco8_yaml(root)

            self.assertTrue(data_yaml.exists())
            self.assertIn("person", data_yaml.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
