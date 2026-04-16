import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from analysis.preprocessing import Loader, SessionBuilder
from analysis.features import SessionFeatureExtractor
from analysis.labeling import AttackLabeler, AutomationProfiler
from analysis.exporter import IntelExporter

DB_PATH = str(_ROOT / "central" / "database" / "honeyiot_central.db")
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
OUT_PARQUET = str(OUTPUT_DIR / "session_intelligence.parquet")
OUT_JSON = str(OUTPUT_DIR / "session_intelligence.json")


def main():
    t0 = time.time()
    print("=" * 60)
    print("  Honeypot Behavioral Intelligence Engine")
    print("=" * 60)

    loader = Loader(DB_PATH)
    events = loader.load()

    builder = SessionBuilder(gap_minutes=30)
    sessions = builder.build(events)

    extractor = SessionFeatureExtractor()
    sessions = extractor.extract_all(sessions)

    labeler = AttackLabeler()
    sessions = labeler.label(sessions)

    profiler = AutomationProfiler()
    sessions = profiler.profile(sessions)

    exporter = IntelExporter()
    exporter.save_parquet(sessions, OUT_PARQUET)
    exporter.save_json(sessions, OUT_JSON)

    elapsed = time.time() - t0
    print("=" * 60)
    print(f"  Completed: {elapsed:.1f}s")
    print(f"  {len(sessions):,} sessions, {len(sessions.columns)} columns")
    print(f"  Parquet: {OUT_PARQUET}")
    print(f"  JSON:    {OUT_JSON}")
    print("=" * 60)


if __name__ == "__main__":
    main()
