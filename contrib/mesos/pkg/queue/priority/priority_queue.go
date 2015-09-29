/*
Copyright 2015 The Kubernetes Authors All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package priority

// Queue is a priority queue of Items that implements container/heap.Interface.
type Queue []Item

func NewPriorityQueue() *Queue {
	var queue Queue
	return &queue
}

func (pq Queue) Len() int { return len(pq) }

func (pq Queue) Less(i, j int) bool {
	return pq[i].Priority().Before(pq[j].Priority())
}

func (pq Queue) Swap(i, j int) {
	pq[i], pq[j] = pq[j], pq[i]
	pq[i].SetIndex(i)
	pq[j].SetIndex(j)
}

func (pq *Queue) Push(x interface{}) {
	n := len(*pq)
	item := x.(Item)
	item.SetIndex(n)
	*pq = append(*pq, item)
}

func (pq *Queue) Pop() interface{} {
	old := *pq
	n := len(old)
	item := old[n-1]
	item.SetIndex(-1) // for safety
	*pq = old[0 : n-1]
	return item
}
