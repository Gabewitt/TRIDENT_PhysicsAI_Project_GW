from typing import TypedDict
from pathlib import Path
from typing import Union, List
import torch
import pandas as pd
from torch.utils.data import Dataset


class TridentData(TypedDict):
    
    features: torch.Tensor
    mask: torch.Tensor
    truth: torch.Tensor


class TridentDataset(Dataset):
    
    def __init__(
        self,
        feature_path: Union[Path, str],
        truth_path:   Union[Path, str],
        truth_keys:   Union[List[str], str] = "initial_state_energy",
        maximum_length: int = 64,
    ) -> None:
       
        self._feature_path = feature_path
        self._truth_path = truth_path
        self._truth_keys = truth_keys
        self.maximum_length = maximum_length

        self._features = torch.from_numpy(
            pd.read_parquet(feature_path).values
        ).to(torch.float32)
        self._truth = torch.from_numpy(
            pd.read_parquet(truth_path)[truth_keys].values
        ).to(torch.float32)
        self._cumulative_lengths = torch.from_numpy(
            pd.read_parquet(truth_path)["cumulative_lengths"].values
        ).to(torch.int64)

    @property
    def features(self) -> torch.Tensor:
       
        return self._features

    @property
    def truth(self) -> torch.Tensor:
   
        return self._truth

    @property
    def cumulative_lengths(self) -> torch.Tensor:
        
        return self._cumulative_lengths

    def __getitem__(self, index: int) -> TridentData:
        
        if index == 0:
            start = 0
        else:
            start = self.cumulative_lengths[index - 1]
        end = self.cumulative_lengths[index]

        event_feature = self.features[start:end]
        mask = torch.zeros(self.maximum_length)
        mask[:len(event_feature)] = 1

        # Pad the features to the maximum length
        event_feature = torch.nn.functional.pad(
            event_feature,
            (0, 0, 0, self.maximum_length - len(event_feature)),
            value=0,
        )

        return {
            "features": event_feature,
            "mask": mask,
            "truth": self.truth[index],
        }

    def __len__(self) -> int:
       
        return len(self.cumulative_lengths)

    def __str__(self) -> str:

        return (
            f"TridentDataset("
            f"\n  number of events: {len(self)},"
            f"\n  number of features: {self.features.shape[1]},"
            f"\n  Number of pulses: {self.cumulative_lengths[-1]},"
            f"\n  maximum length: {self.maximum_length},"
            f"\n  truth keys: {self._truth_keys},"
            f"\n)"
        )

    def __repr__(self) -> str:

        return (
            f"TridentDataset("
            f"\n  feature_path={self._feature_path},"
            f"\n  truth_path={self._truth_path},"
            f"\n  truth_keys={self._truth_keys},"
            f"\n  maximum_length={self.maximum_length},"
            f"\n)"
        )
