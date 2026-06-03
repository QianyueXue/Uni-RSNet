import os
import json
import time
import random
from typing import Iterator, List, Dict, Any, Optional
from torch.utils.data import Sampler


def _get_rank_default() -> int:
    for k in ("RANK", "LOCAL_RANK", "OMPI_COMM_WORLD_RANK"):
        v = os.environ.get(k, None)
        if v is not None:
            try:
                return int(v)
            except Exception:
                pass
    return 0


# by xueqianyue
class SigBatchSampler(Sampler[List[int]]):


    def __init__(
        self,
        dataset: Any,
        batch_size: int,
        drop_last: bool = False,
        shuffle: bool = True,
        same_video: bool = True,
        seed: int = 0,
        log_file: Optional[str] = None,


        head_random: bool = True,
        head_n: int = 4,
    ) -> None:
        super().__init__(dataset)
        assert hasattr(dataset, "__len__"), "SigBatchSampler 需要 map-style dataset（实现 __len__）"
        assert batch_size > 0, "batch_size 必须 > 0"
        assert head_n > 0, "head_n 必须 > 0"
        assert batch_size >= head_n, f"batch_size({batch_size}) 必须 ≥ head_n({head_n})"

        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.drop_last = bool(drop_last)
        self.shuffle = bool(shuffle)
        self.same_video = bool(same_video)
        self.seed = int(seed)
        self.epoch = 0

        self.head_random = bool(head_random)
        self.head_n = int(head_n)


        self.rank = _get_rank_default()
        if log_file:
            try:
                self.log_file = log_file.format(rank=self.rank)
            except Exception:
                self.log_file = log_file
            os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        else:
            self.log_file = None


        self._bucket_keys: List[Optional[Any]] = self._build_bucket_keys()
        self._diagnosed = False


        n = len(self.dataset)
        if self.drop_last:
            self._num_batches = n // self.batch_size
        else:
            self._num_batches = (n + self.batch_size - 1) // self.batch_size

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self._num_batches

    def __iter__(self) -> Iterator[List[int]]:
        rng = random.Random(self.seed ^ (self.epoch + 0x9E3779B97F4A7C15))


        if not self._diagnosed:
            uniq = {k for k in self._bucket_keys if k is not None}
            if self.same_video:
                if len(uniq) == 0:
                    print("[SigBatchSampler][WARN] same_video=True 但未探测到有效桶键（如 video_id），"
                          "将退化为完全随机。")
                else:
                    print(f"[SigBatchSampler][INFO] 检测到 {len(uniq)} 个桶（视频）。"
                          f" head_random={self.head_random}, head_n={self.head_n}, batch_size={self.batch_size}")
            self._diagnosed = True


        if not (self.same_video and any(k is not None for k in self._bucket_keys) and self.head_random):

            all_indices = list(range(len(self.dataset)))
            if self.shuffle:
                rng.shuffle(all_indices)
            i = 0
            batch_id = 0
            while i + self.batch_size <= len(all_indices):
                b = all_indices[i:i + self.batch_size]
                self._log_batch(b, batch_id); batch_id += 1
                yield b
                i += self.batch_size
            return


        buckets = self._group_by_bucket()
        keys = list(buckets.keys())
        if self.shuffle:
            rng.shuffle(keys)
        for k in keys:
            if self.shuffle:
                rng.shuffle(buckets[k])


        head_chunks: List[List[int]] = []
        random_pool: List[int] = []
        for k in keys:
            idxs = buckets[k]
            full = (len(idxs) // self.head_n) * self.head_n

            for i in range(0, full, self.head_n):
                head_chunks.append(idxs[i:i + self.head_n])

            rem = idxs[full:]
            if rem:
                random_pool.extend(rem)

        if self.shuffle:
            rng.shuffle(head_chunks)
            rng.shuffle(random_pool)

        used = set()
        pool = list(random_pool)
        all_indices = set(range(len(self.dataset)))


        batch_id = 0
        out_batches: List[List[int]] = []


        need_tail = self.batch_size - self.head_n
        for head in head_chunks:

            if any(i in used for i in head):
                continue


            for i in head:
                used.add(i)


            if self.shuffle:
                rng.shuffle(pool)
            if len(pool) < need_tail:


                if self.drop_last:

                    for i in head:
                        used.remove(i)
                    continue
                else:


                    for i in head:
                        used.remove(i)
                    break

            tail = pool[:need_tail]
            del pool[:need_tail]
            for i in tail:
                used.add(i)

            b = list(head) + list(tail)
            self._log_batch(b, batch_id); batch_id += 1
            out_batches.append(b)


            if len(out_batches) >= self._num_batches:
                break


        if len(out_batches) < self._num_batches:

            remaining = [i for i in range(len(self.dataset)) if i not in used]
            if self.shuffle:
                rng.shuffle(remaining)

            i = 0
            while len(out_batches) < self._num_batches and i + self.batch_size <= len(remaining):
                b = remaining[i:i + self.batch_size]
                i += self.batch_size
                for j in b:
                    used.add(j)
                self._log_batch(b, batch_id); batch_id += 1
                out_batches.append(b)


        for b in out_batches:
            yield b


    def _group_by_bucket(self) -> Dict[Optional[Any], List[int]]:

        n = len(self.dataset)
        keys = self._bucket_keys
        buckets: Dict[Optional[Any], List[int]] = {}
        for idx, k in enumerate(keys):
            buckets.setdefault(k, []).append(idx)
        return buckets

    def _build_bucket_keys(self) -> List[Optional[Any]]:
        n = len(self.dataset)
        ds = self.dataset

        if self.same_video and hasattr(ds, "get_video_id") and callable(getattr(ds, "get_video_id")):
            try:
                return [ds.get_video_id(i) for i in range(n)]
            except Exception:
                pass

        if self.same_video and hasattr(ds, "video_ids"):
            try:
                vids = getattr(ds, "video_ids")
                if isinstance(vids, (list, tuple)) and len(vids) == n:
                    return list(vids)
            except Exception:
                pass

        if self.same_video and hasattr(ds, "coco") and hasattr(ds, "ids"):
            try:
                coco = getattr(ds, "coco")
                ids = list(getattr(ds, "ids"))
                keys = []
                for img_id in ids:
                    info = coco.loadImgs([img_id])[0]
                    keys.append(info.get("video_id", None))
                if any(k is not None for k in keys):
                    return keys
            except Exception:
                pass


        return [None] * n

    def _resolve_name(self, idx: int) -> str:
        ds = self.dataset
        try:
            if hasattr(ds, "coco") and hasattr(ds, "ids"):
                img_id = ds.ids[idx]
                info = ds.coco.loadImgs([img_id])[0]
                fn = info.get("file_name", None)
                if fn:
                    return str(fn)
        except Exception:
            pass
        for attr in ("samples", "images", "imgs", "paths"):
            if hasattr(ds, attr):
                try:
                    entry = getattr(ds, attr)[idx]
                    if isinstance(entry, (list, tuple)) and entry and isinstance(entry[0], (str, os.PathLike)):
                        return os.fspath(entry[0])
                    if isinstance(entry, dict):
                        for k in ("file_name", "filename", "path", "name"):
                            if k in entry:
                                return str(entry[k])
                except Exception:
                    pass
        return str(idx)

    def _log_batch(self, batch_indices: List[int], batch_id: int) -> None:
        if not self.log_file:
            return
        try:
            rec = {
                "ts": int(time.time()),
                "epoch": self.epoch,
                "rank": self.rank,
                "batch_id": batch_id,
                "indices": batch_indices,
                "files": [self._resolve_name(i) for i in batch_indices],
            }
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"[SigBatchSampler][WARN] 写日志失败: {e}")
