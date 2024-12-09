from typing import List, Optional, Union, Iterator, Callable, Any, TypeVar, Generic
from collections.abc import Sequence
from dataclasses import dataclass
import heapq
from contextlib import contextmanager

T = TypeVar('T')

@dataclass
class HeapNode(Generic[T]):
    priority: Union[int, float]
    value: T
    insertion_count: int
    
    def __lt__(self, other: 'HeapNode') -> bool:
        if self.priority == other.priority:
            return self.insertion_count < other.insertion_count
        return self.priority < other.priority

class MinHeap(Generic[T]):
    def __init__(self, 
                 items: Optional[Sequence[Union[int, tuple[Union[int, float], T]]]] = None,
                 key: Optional[Callable[[T], Union[int, float]]] = None):
        self.heap: List[HeapNode[T]] = []
        self.size: int = 0
        self._insertion_count: int = 0
        self.key_func = key
        self._snapshots: List[List[HeapNode[T]]] = []
        
        if items:
            self._build_from_items(items)

    def _build_from_items(self, items: Sequence[Union[int, tuple[Union[int, float], T]]]) -> None:
        for item in items:
            if isinstance(item, (int, float)):
                self.push(item)
            else:
                priority, value = item
                self.push_with_priority(value, priority)
    
    def push(self, value: T) -> None:
        priority = self.key_func(value) if self.key_func else value
        self.push_with_priority(value, priority)
    
    def push_with_priority(self, value: T, priority: Union[int, float]) -> None:
        node = HeapNode(priority, value, self._insertion_count)
        self._insertion_count += 1
        heapq.heappush(self.heap, node)
        self.size += 1
    
    def pop(self) -> Optional[T]:
        if not self.heap:
            return None
        self.size -= 1
        return heapq.heappop(self.heap).value
    
    def peek(self) -> Optional[tuple[Union[int, float], T]]:
        if not self.heap:
            return None
        node = self.heap[0]
        return (node.priority, node.value)

    def bulk_push(self, items: Sequence[T]) -> None:
        for item in items:
            self.push(item)
        self._heapify()
    
    def merge(self, other: 'MinHeap[T]') -> None:
        self.heap.extend(other.heap)
        self.size += other.size
        self._heapify()
    
    def is_empty(self) -> bool:
        return self.size == 0
    
    def drain(self) -> Iterator[T]:
        while not self.is_empty():
            yield self.pop()

    @contextmanager
    def snapshot(self):
        self._snapshots.append(self.heap[:])
        try:
            yield
        finally:
            self.heap = self._snapshots.pop()
            self.size = len(self.heap)

    def k_smallest(self, k: int) -> List[T]:
        with self.snapshot():
            return [self.pop() for _ in range(min(k, self.size))]

    def update_priority(self, value: T, new_priority: Union[int, float]) -> bool:
        for i, node in enumerate(self.heap):
            if node.value == value:
                old_priority = node.priority
                self.heap[i].priority = new_priority
                if new_priority < old_priority:
                    self._sift_up(i)
                else:
                    self._sift_down(i)
                return True
        return False

    def remove(self, value: T) -> bool:
        for i, node in enumerate(self.heap):
            if node.value == value:
                self.heap[i] = self.heap[-1]
                self.heap.pop()
                self.size -= 1
                if i < len(self.heap):
                    self._sift_down(i)
                return True
        return False
    
    def _heapify(self) -> None:
        heapq.heapify(self.heap)
    
    def _sift_up(self, idx: int) -> None:
        temp = self.heap[idx]
        while idx > 0:
            parent_idx = (idx - 1) // 2
            if self.heap[parent_idx] <= temp:
                break
            self.heap[idx] = self.heap[parent_idx]
            idx = parent_idx
        self.heap[idx] = temp
    
    def _sift_down(self, idx: int) -> None:
        temp = self.heap[idx]
        while True:
            smallest = idx
            left = 2 * idx + 1
            right = 2 * idx + 2
            
            if left < self.size and self.heap[left] < temp:
                smallest = left
            if right < self.size and self.heap[right] < self.heap[smallest]:
                smallest = right
                
            if smallest == idx:
                break
                
            self.heap[idx] = self.heap[smallest]
            idx = smallest
        self.heap[idx] = temp

    def __len__(self) -> int:
        return self.size

    def __iter__(self) -> Iterator[T]:
        return iter(node.value for node in self.heap)

if __name__ == "__main__":
    heap = MinHeap[tuple[str, int]]()
    items = [
        (5, ("task1", 1)),
        (2, ("task2", 2)),
        (8, ("task3", 3)),
        (1, ("task4", 4)),
    ]
    
    for priority, value in items:
        heap.push_with_priority(value, priority)

    assert heap.peek()[1][0] == "task4"
    smallest = heap.k_smallest(2)
    assert len(smallest) == 2
    assert smallest[0][0] == "task4"
    assert smallest[1][0] == "task2"

    with heap.snapshot():
        heap.pop()
        heap.pop()
        assert len(heap) == 2
    assert len(heap) == 4

    heap.update_priority(("task3", 3), 0)
    assert heap.peek()[1][0] == "task3"

    other_heap = MinHeap[tuple[str, int]]()
    other_heap.push_with_priority(("task5", 5), 3)
    heap.merge(other_heap)
    assert len(heap) == 5

    new_items = [("task6", 6), ("task7", 7)]
    heap.bulk_push(new_items)
    assert len(heap) == 7

    sorted_items = list(heap.drain())
    assert len(sorted_items) == 7
    assert heap.is_empty()
