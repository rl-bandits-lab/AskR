import random
from typing import Dict, List
from dataclasses import dataclass, asdict


@dataclass
class SubQAEntry:
    """A single SubQ-SubA pair with metadata."""
    sub_question: str
    sub_answer: str
    reward: float
    position: int
    trajectory_id: str = ""

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class TrajectoryEntry:
    """A SubQA trajectory and its intrinsic reward (Eq. 4)."""
    trajectory_id: str
    subqa_entries: List[SubQAEntry]
    avg_reward: float


class SegmentedBuffer:
    """Buffer storing SubQ-SubA pairs by position (opening/middle/ending)."""

    def __init__(self, threshold: float = 0.5, max_per_segment: int = 3):
        self.threshold = threshold
        self.max_per_segment = max_per_segment
        self.opening: List[SubQAEntry] = []
        self.middle: List[SubQAEntry] = []
        self.ending: List[SubQAEntry] = []
        self.history: List[Dict] = []
        self.best_reward: float = 0.0

    def _segment_trajectory(self, trajectory: TrajectoryEntry) -> Dict[str, List[SubQAEntry]]:
        """Segment entries into opening/middle/ending."""
        entries = trajectory.subqa_entries
        n = len(entries)
        segmented = {'opening': [], 'middle': [], 'ending': []}

        if n == 0:
            return segmented
        elif n == 1:
            segmented['opening'].append(entries[0])
        elif n == 2:
            segmented['opening'].append(entries[0])
            segmented['ending'].append(entries[1])
        else:
            third = n // 3
            remainder = n % 3
            opening_end = third + (1 if remainder > 0 else 0)
            ending_start = n - third - (1 if remainder > 1 else 0)

            for i, entry in enumerate(entries):
                if i < opening_end:
                    segmented['opening'].append(entry)
                elif i >= ending_start:
                    segmented['ending'].append(entry)
                else:
                    segmented['middle'].append(entry)

        return segmented

    def add_trajectory(self, trajectory: TrajectoryEntry) -> None:
        """Add trajectory to buffer if meets threshold."""
        if trajectory.avg_reward < self.threshold:
            return

        if trajectory.avg_reward > self.best_reward:
            self.best_reward = trajectory.avg_reward

        segmented = self._segment_trajectory(trajectory)
        for seg_name, seg_items in segmented.items():
            segment_list = getattr(self, seg_name)
            for item in seg_items:
                item.reward = trajectory.avg_reward
                segment_list.append(item)
            seen = set()
            deduped = []
            for item in segment_list:
                key = (item.sub_question[:100].lower().strip(), item.sub_answer[:100].lower().strip())
                if key not in seen:
                    seen.add(key)
                    deduped.append(item)
            deduped = sorted(deduped, key=lambda x: x.reward, reverse=True)[:self.max_per_segment]
            setattr(self, seg_name, deduped)

    def get_prompt_context(self) -> str:
        """Prompt context with one randomly sampled SubQA per segment."""
        parts = []
        segment_labels = {
            'opening': 'Opening (initial guiding questions)',
            'middle': 'Middle (intermediate guiding questions)',
            'ending': 'Ending (final guiding questions)'
        }

        for seg_name, label in segment_labels.items():
            items = getattr(self, seg_name)
            if items:
                sampled = random.choice(items)
                parts.append(f"### {label}:\nQ: {sampled.sub_question}\nA: {sampled.sub_answer}")
            else:
                parts.append(f"### {label}:\n(No good examples collected yet)")

        return "\n\n".join(parts)

    def get_state(self) -> Dict:
        return {
            'opening': [item.to_dict() for item in self.opening],
            'middle': [item.to_dict() for item in self.middle],
            'ending': [item.to_dict() for item in self.ending]
        }

    def save_history_snapshot(self, iteration_id: int):
        self.history.append({'iteration': iteration_id, 'state': self.get_state()})
