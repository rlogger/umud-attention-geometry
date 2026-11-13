import numpy as np
import pandas as pd
from PIL import Image
import pytest

from umud_attention.workflow import prepare_dataset


def _dataset(root):
    rng = np.random.default_rng(82)
    for task in ('apo', 'fasc'):
        for role in ('imgs', 'masks'):
            (root/f'{task}_{role}_v1').mkdir(parents=True)
        for i in range(4):
            pixels = rng.integers(0, 255, (24, 32), dtype=np.uint8)
            mask = np.zeros((24, 32), dtype=np.uint8)
            if i != 3:
                mask[8:10, 4:25] = 255
            Image.fromarray(pixels).save(root/f'{task}_imgs_v1'/f'image_{i}.tif')
            Image.fromarray(mask).save(root/f'{task}_masks_v1'/f'image_{i}.tif')


def test_uniform_annotations_are_recorded_and_excluded(tmp_path):
    root = tmp_path/'data'; _dataset(root)
    train, _, _, report = prepare_dataset(root, tmp_path/'run')
    assert len(train) == 6
    assert report['excluded_uniform_masks'] == 2
    assert report['raw_task_counts'] == {'apo':4, 'fasc':4}
    excluded = pd.read_csv(tmp_path/'run'/'excluded_annotations.csv')
    assert set(excluded.image_id) == {'image_3.tif'}
    assert train.groupby('task').fold.nunique().eq(3).all()


def test_strict_uniform_policy_stops_before_training(tmp_path):
    root = tmp_path/'data'; _dataset(root)
    with pytest.raises(ValueError, match='uniform masks'):
        prepare_dataset(root, tmp_path/'run', empty_mask_policy='error')
    assert (tmp_path/'run'/'excluded_annotations.csv').exists()


def test_reviewed_groups_for_excluded_rows_are_removed(tmp_path):
    root = tmp_path/'data'; _dataset(root)
    groups = pd.DataFrame([{'task':'apo','image_id':'image_3.tif','group':'empty'}])
    train, _, _, report = prepare_dataset(root, tmp_path/'run', groups_csv=groups)
    assert len(train) == 6 and report['excluded_uniform_masks'] == 2
