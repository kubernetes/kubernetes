/*
   Copyright 2014-2021 Docker Inc.

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

package spdystream

import (
	"container/heap"
	"sync"

	"github.com/moby/spdystream/spdy"
)

type prioritizedFrame struct {
	frame    spdy.Frame
	priority uint8
	insertId uint64
}

type frameQueue []*prioritizedFrame

func (fq frameQueue) Len() int {
	return len(fq)
}

func (fq frameQueue) Less(i, j int) bool {
	if fq[i].priority == fq[j].priority {
		return fq[i].insertId < fq[j].insertId
	}
	return fq[i].priority < fq[j].priority
}

func (fq frameQueue) Swap(i, j int) {
	fq[i], fq[j] = fq[j], fq[i]
}

func (fq *frameQueue) Push(x interface{}) {
	*fq = append(*fq, x.(*prioritizedFrame))
}

func (fq *frameQueue) Pop() interface{} {
	old := *fq
	n := len(old)
	*fq = old[0 : n-1]
	return old[n-1]
}

type PriorityFrameQueue struct {
	queue        *frameQueue
	c            *sync.Cond
	size         int
	nextInsertId uint64
	drain        bool
}

func NewPriorityFrameQueue(size int) *PriorityFrameQueue {
	queue := make(frameQueue, 0, size)
	heap.Init(&queue)

	return &PriorityFrameQueue{
		queue: &queue,
		size:  size,
		c:     sync.NewCond(&sync.Mutex{}),
	}
}

func (q *PriorityFrameQueue) Push(frame spdy.Frame, priority uint8) {
	q.c.L.Lock()
	defer q.c.L.Unlock()
	for q.queue.Len() >= q.size {
		q.c.Wait()
	}
	pFrame := &prioritizedFrame{
		frame:    frame,
		priority: priority,
		insertId: q.nextInsertId,
	}
	q.nextInsertId = q.nextInsertId + 1
	heap.Push(q.queue, pFrame)
	q.c.Signal()
}

func (q *PriorityFrameQueue) Pop() spdy.Frame {
	q.c.L.Lock()
	defer q.c.L.Unlock()
	for q.queue.Len() == 0 {
		if q.drain {
			return nil
		}
		q.c.Wait()
	}
	frame := heap.Pop(q.queue).(*prioritizedFrame).frame
	q.c.Signal()
	return frame
}

func (q *PriorityFrameQueue) Drain() {
	q.c.L.Lock()
	defer q.c.L.Unlock()
	q.drain = true
	q.c.Broadcast()
}
