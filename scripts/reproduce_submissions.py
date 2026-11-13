"""Reconstruct the submitted predictions from saved measurements and segments."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parents[1]
TARGETS = ('pa_deg', 'fl_mm', 'mt_mm')
SCHEMA = ('image_id', *TARGETS)


def read_csv(path):
    with path.open(newline='') as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
    if len(rows) != 309 or len({r['image_id'] for r in rows}) != 309:
        raise ValueError(f'Expected 309 unique IDs: {path}')
    return reader.fieldnames, rows


def trimmed_mean(values):
    median = statistics.median(values)
    mad = statistics.median(abs(value - median) for value in values)
    kept = [value for value in values if value == median] if mad == 0 else [
        value for value in values if abs(value - median) <= 3 * 1.4826 * mad]
    return statistics.mean(kept)


def verify_hash(path, expected):
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise ValueError(f'File checksum mismatch: {path}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT/'reproduced')
    args = parser.parse_args()
    scores = json.loads((ROOT/'results/scores.json').read_text())
    for record in scores:
        verify_hash(ROOT/record['file'], record['sha256'])
    reference_path = ROOT/'references/colton_blend34.csv'
    verify_hash(reference_path, '7f155ec03ac1d5b94353039ded4c9a9b00f1022806d0fc00aad6bc0926eb3ca6')
    header, first = read_csv(ROOT/'submissions/attention_hybrid20.csv')
    second_header, second = read_csv(ROOT/'submissions/attention_hybrid20_trimmed_fl.csv')
    reference_header, reference = read_csv(reference_path)
    _, measurements = read_csv(ROOT/'results/measurements_partial.csv')
    _, quality = read_csv(ROOT/'results/quality.csv')
    if tuple(header) != SCHEMA or second_header != header or reference_header != header:
        raise ValueError('Unexpected submission columns')
    ids = [r['image_id'] for r in first]
    for rows in (second, reference, measurements, quality):
        if {r['image_id'] for r in rows} != set(ids):
            raise ValueError('Input image IDs differ')
    reference = {r['image_id']: r for r in reference}
    measurements = {r['image_id']: r for r in measurements}
    eligible = {r['image_id'] for r in quality if r['eligible_for_blend'] == 'True'}
    if len(eligible) != 102:
        raise ValueError('Expected the original 102 eligible images')

    first_rebuilt = []
    max_first_error = 0.0
    for row in first:
        image_id = row['image_id']
        rebuilt = {'image_id': image_id}
        for target in TARGETS:
            base = float(reference[image_id][target])
            value = (.8 * base + .2 * float(measurements[image_id][target])
                     if image_id in eligible else base)
            error = abs(value - float(row[target]))
            max_first_error = max(max_first_error, error)
            if not math.isfinite(value) or error > 1e-10:
                raise ValueError(f'First submission differs: {image_id} {target}')
            rebuilt[target] = repr(value)
        first_rebuilt.append(rebuilt)

    replay = json.loads((ROOT/'results/trimmed_fl_replay.json').read_text())['items']
    replay = {r['image_id']: r for r in replay}
    if set(replay) != eligible:
        raise ValueError('Replay must contain exactly the original eligible images')
    second_rebuilt = []
    for row in first:
        row = dict(row)
        image_id = row['image_id']
        if image_id in eligible:
            item = replay[image_id]
            variants = [trimmed_mean(v['accepted_segment_lengths_mm'])
                        for v in item['variants'] if v['status'] == 'ok']
            if len(variants) < 2 or max(variants) - min(variants) > 12:
                raise ValueError(f'Unstable FL: {image_id}')
            fl = statistics.median(variants)
            if abs(fl - item['trimmed_fl_mm']) > 1e-10:
                raise ValueError(f'Recorded aggregation differs: {image_id}')
            row['fl_mm'] = repr(.8 * float(reference[image_id]['fl_mm']) + .2 * fl)
        second_rebuilt.append(row)
    if second_rebuilt != second:
        raise ValueError('Second submission differs from reconstructed values')

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in [('attention_hybrid20.csv', first_rebuilt),
                       ('attention_hybrid20_trimmed_fl.csv', second_rebuilt)]:
        destination = args.output_dir/name
        if destination.resolve() in {ROOT/r['file'] for r in scores}:
            raise ValueError('Choose an output directory outside submissions/')
        with destination.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=header)
            writer.writeheader()
            writer.writerows(rows)
    verify_hash(args.output_dir/'attention_hybrid20_trimmed_fl.csv', scores[1]['sha256'])
    print(json.dumps({'rows':309, 'eligible':102, 'reference_only':207,
                      'first_max_numeric_difference':max_first_error,
                      'second_byte_identical':True,
                      'output_dir':str(args.output_dir)}, indent=2))


if __name__ == '__main__':
    main()
