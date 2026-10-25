# Ensemble Classifier for Data Streams

Coursework for **COMPX523-26A: Machine Learning for Data Streams**
(University of Waikato). Builds an online ensemble classifier that
learns continuously from a non-stationary stream with class imbalance
and concept drift, under a hard 250 KB memory budget.

> **Post-mortem included.** The originally shipped configuration had a
> memory-enforcement bug that this portfolio revision documents and
> corrects. See [§ The memory bug](#the-memory-bug-and-the-fix) below.

## Problem

The stream is the instructor's synthetic RBF generator mixture
configured to mimic one year of sensor data:

| Feature | Value |
| --- | --- |
| Instances | ~105 000 (5-minute samples over 12 months) |
| Features | 12 numerical (8 relevant, 4 irrelevant) |
| Classes | 2 |
| Imbalance | evolves between 2 : 1 and 8 : 1 |
| Drifts | 1 abrupt + 3 gradual (over days, a week, a month) |
| Memory budget | **≤ 250 KB**, enforced server-side |

The classifier has to ingest one instance at a time, predict, then
train, never holding more than 250 KB of state.

## Architecture

Two classifiers, both built on [CapyMOA][capymoa] (Python bindings for
[MOA][moa]):

| File | Role | Default `n` |
| --- | --- | --- |
| [`ensemble_classifier.py`](ensemble_classifier.py) | Spec-literal: Incremental GNB + ADWIN + Online Bagging + Leveraged Bagging | 10 |
| [`champion_classifier.py`](champion_classifier.py) | Tournament: Hoeffding Tree + ADWIN + same bagging logic | 25 |

Shared scaffolding:

- **Online Bagging:** per incoming instance, each base learner
  receives weight `k ~ Poisson(λ)` and is trained `k` times. Bagging
  weights are drawn vectorised in NumPy.
- **ADWIN drift detection:** one detector per ensemble member,
  fed the 0/1 error stream. On `detected_change()` the member's base
  learner is re-initialised. The champion uses an `_ADWINNoHistory`
  subclass that skips the unbounded `MOADriftDetector.data` bookkeeping
  list.
- **Leveraged Bagging:** a label ring buffer (numpy `int8`)
  estimates the live class distribution. When the incoming instance is
  minority-side of a threshold, λ scales proportionally to the
  imbalance ratio (capped) so the minority is oversampled in the bagging
  draw. Ensemble uses 40 % / cap 12, champion uses 35 % / cap 8.

The champion swaps the GNB base learner (permitted by PDF §3.1) for a
memory-bounded Hoeffding Tree because GNB's single-Gaussian-per-class
assumption is violated by the RBF generator's mixture decision surface.

## The memory bug and the fix

The submitted version reported **156 KB of memory** via
`pympler.asizeof`. The instructor's evaluator, measuring the JVM heap,
reported **~25 MB** and applied the over-limit penalty. Two bugs
combined to produce that gap; both are fixed in this revision.

### Bug 1: `pympler.asizeof` does not see MOA's Java heap

`CapyMOA.HoeffdingTree` is a thin Python wrapper over a Java object
reached through JPype. `asizeof` walks the Python object graph and
reports the wrapper, *not* the JVM heap that holds the actual node
arrays, attribute observers, and split statistics.

This revision adds `ChampionClassifier.measure_java_byte_size()`,
which sums MOA's own `measureByteSize()` across every base learner.
That is the number the instructor's evaluator sees and the number
`max_byte_size` is enforced against.

### Bug 2: `memory_estimate_period` defaults to 1 000 000

MOA only checks each tree's actual size against `max_byte_size` every
`memory_estimate_period` instances. CapyMOA's default value is **one
million**. On the assignment's 37 K sample stream and 100 K synthetic
stream the check never fires, so `max_byte_size=2*1024` is silently
ignored and trees grow to whatever the data triggers.

This revision passes `memory_estimate_period=1000`, which makes
enforcement effectively continuous at negligible runtime cost.

### Bug 3: without MOA's size agent, nothing is enforced

Found while re-running this revision in October 2026. MOA's
`measureByteSize()` relies on its `SizeOfAgent` Java agent. When the JVM
starts without it, the call returns 0, so each tree believes it is empty,
`max_byte_size` is never enforced, and `measure_java_byte_size()` reports
nothing. Run with the agent loaded: `make_sizeof_agent.py` builds a small
agent jar from CapyMOA's own `moa.jar`, and `run_evaluation.py` now warns
when MOA measures 0 bytes.

### How the fix looks

```python
HoeffdingTree(
    schema=self._schema,
    random_seed=seed,
    max_byte_size=self.max_byte_size,          # 8 KB per tree
    memory_estimate_period=self.memory_estimate_period,  # 1000  <-- the missing piece
    grace_period=self.grace_period,
    confidence=self.confidence,
    stop_mem_management=True,
)
```

Defaults were also resized so the realistic per-tree budget fits the
ensemble: `n_members=25, max_byte_size=8 KB` gives ≈ 200 KB worst-case
trees plus ≈ 30 KB ADWIN/overhead. The instructor's own
HoeffdingAdaptiveTree(n=10) reference classifier landed 0.88 TScore at
112 KB total, which validates the regime.

## Results after the fix

On the synthetic RBF stream (100,000 instances, 4 drifts) with MOA's size agent loaded, the fixed champion (n=25, 8 KB per tree, memory_estimate_period=1000) reaches a tournament score of 0.9048, with its trees measured at 204.8 KB on the Java side and about 234 KB in total, under the 250 KB budget.

The instructor's sample stream is course material and was not available for
this re-run, so `results.csv` holds the synthetic rows only.
`results-pre-fix.csv` keeps the submitted configuration's results (n=100,
2 KB per tree, no effective enforcement), which broke the memory limit; its
higher scores came from trees that were never capped. A run without the size
agent shows the same effect: the cap is silently ignored and the score rises.

## Run

```bash
cd streaming-ensemble-classifier
python3 -m venv .venv
source .venv/bin/activate
pip install capymoa pympler numpy scipy

export JAVA_HOME=/path/to/openjdk-17  # CapyMOA 0.13 needs Java 17
python3 make_sizeof_agent.py           # lets MOA measure, and so cap, tree size
export JAVA_TOOL_OPTIONS="-javaagent:$PWD/sizeof-agent.jar"
python3 run_evaluation.py
```

`run_evaluation.py` runs three classifiers (NaiveBayes baseline, the
spec-literal ensemble, the champion) on both the released sample CSV
and a synthetic RBF stream, and prints prequential metrics plus the
Python-side and Java-side memory breakdown side by side.

## Files

```
streaming-ensemble-classifier/
├── README.md                ← this file
├── REPORT.md / REPORT.pdf   ← submission write-up (with post-mortem)
├── ensemble_classifier.py   ← Programming Task (spec-literal GNB ensemble)
├── champion_classifier.py   ← Tournament Champion (Hoeffding-Tree ensemble)
├── ht_experiment.py         ← Hoeffding Tree experiments
├── run_evaluation.py        ← prequential evaluation harness
├── quick_test.py            ← fast smoke test
├── make_sizeof_agent.py     ← builds MOA's size agent jar (see Bug 3)
├── results.csv              ← results of this revision (synthetic stream)
└── results-pre-fix.csv      ← results of the submitted, over-limit configuration
```

> **Data not included.** The instructor-released sample stream
> (`sample_test_csv.csv` / `.arff`) and the assignment brief are course
> material and are not redistributed here. Point `run_evaluation.py` at
> your own stream path to reproduce.

## What I would do differently

- Verify memory via **both** the language the assignment hints at
  (`pympler.asizeof`) **and** the language MOA uses internally
  (`measureByteSize()`). Whichever number is bigger is the truth.
- When the spec hints at a memory cap, instrument the run on a stream
  long enough (`>> memory_estimate_period`) to trigger the enforcement
  pass. If you can't observe enforcement firing, you're not measuring
  what you think you are.
- Check that the measurement itself works: a size of 0 bytes is a
  missing agent, not a small model.

## License

MIT (see `LICENSE`). The course data is not included and is not covered by
the licence.

## Stack

CapyMOA 0.13 (Python 3.13, JPype, OpenJDK 17), MOA (Java), NumPy,
SciPy, Pympler.

[capymoa]: https://capymoa.org/
[moa]: https://moa.cms.waikato.ac.nz/
