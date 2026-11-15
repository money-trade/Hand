"""ทดสอบการเฉลี่ย/โหวตหลายเฟรมก่อนบันทึกข้อความ (python -m unittest discover -s tests -v)"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mudra.nlp import NONE_TOKEN, TranscriptBuilder

NAMES = ["a", "b", "c"]


def feed(tb, frames):
    """frames = [(label, conf, still)] -> รายชื่อโทเคนที่ถูกบันทึก"""
    out = []
    for lab, conf, still in frames:
        t = tb.update(lab, conf, still, hand_present=lab != NONE_TOKEN)
        if t:
            out.append(t[0])
    return out


class VotingTranscript(unittest.TestCase):
    def test_steady_pose_commits_once(self):
        tb = TranscriptBuilder(NAMES)
        self.assertEqual(feed(tb, [("a", 0.9, True)] * 40), ["a"])

    def test_tolerates_occasional_bad_frames(self):
        """มีเฟรมหลุด (มั่นใจต่ำ/ทายผิด/ขยับ) เป็นครั้งคราว ต้องยังบันทึกได้ ไม่ต้องเริ่มนับใหม่"""
        tb = TranscriptBuilder(NAMES)
        frames = [("a", 0.9, True)] * 4 + [("b", 0.9, True), ("a", 0.3, True), ("a", 0.9, False)] + [("a", 0.9, True)] * 20
        self.assertEqual(feed(tb, frames), ["a"])

    def test_brief_pose_not_committed(self):
        tb = TranscriptBuilder(NAMES)
        frames = [("a", 0.9, True)] * 5 + [("b", 0.9, True)] * 5 + [("c", 0.9, True)] * 5
        self.assertEqual(feed(tb, frames), [])

    def test_moving_hand_not_committed(self):
        tb = TranscriptBuilder(NAMES)
        self.assertEqual(feed(tb, [("a", 0.95, False)] * 40), [])

    def test_low_confidence_not_committed(self):
        tb = TranscriptBuilder(NAMES)
        self.assertEqual(feed(tb, [("a", 0.3, True)] * 40), [])

    def test_repeat_needs_hand_removed(self):
        tb = TranscriptBuilder(NAMES)
        gap = [(NONE_TOKEN, 0.9, True)] * 4
        frames = [("a", 0.9, True)] * 20 + gap + [("a", 0.9, True)] * 20
        self.assertEqual(feed(tb, frames), ["a", "a"])
        tb2 = TranscriptBuilder(NAMES)
        self.assertEqual(feed(tb2, [("a", 0.9, True)] * 20 + [("a", 0.9, True)] * 20), ["a"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
