import os
import re
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class DatasetPaths:
    """Configuration for dataset file paths."""
    train_dir: str
    test_dir: str
    train_csv_path: str


def read_text_pair(base_dir: str, article_id: int) -> Tuple[str, str]:
    """Read a pair of texts from the dataset."""
    folder = Path(base_dir) / f"article_{article_id:04d}"
    
    with open(folder / "file_1.txt", "r", encoding="utf-8") as f1:
        text1 = f1.read()
    with open(folder / "file_2.txt", "r", encoding="utf-8") as f2:
        text2 = f2.read()
        
    return text1, text2


def list_test_article_ids(test_dir: str) -> List[int]:
    """Get list of test article IDs."""
    article_folders = [d for d in os.listdir(test_dir) if d.startswith("article_")]
    
    def parse_id(name: str) -> int:
        match = re.match(r"article_(\d{4})\Z", name)
        if not match:
            raise ValueError(f"Unexpected folder name: {name}")
        return int(match.group(1))
        
    return sorted([parse_id(d) for d in article_folders])