/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

package e2e

import (
	"sync"
	"time"
)

type QueueItem struct {
	createTime string
	value      interface{}
}

type QueueItems struct {
	pos   int
	mutex *sync.Mutex
	list  []QueueItem
}

type FifoQueue QueueItems

func (fq *FifoQueue) Push(elem interface{}) {
	fq.mutex.Lock()
	fq.list = append(fq.list, QueueItem{time.Now().String(), elem})
	fq.mutex.Unlock()
}

func (fq *FifoQueue) Pop() QueueItem {
	fq.mutex.Lock()
	var val QueueItem
	if len(fq.list)-1 >= fq.pos {
		val = fq.list[fq.pos]
		fq.pos++
	}
	fq.mutex.Unlock()
	return val
}

func (fq FifoQueue) Len() int {
	return len(fq.list[fq.pos:])
}

func (fq *FifoQueue) First() QueueItem {
	return fq.list[fq.pos]
}

func (fq *FifoQueue) Last() QueueItem {
	return fq.list[len(fq.list)-1]
}

func (fq *FifoQueue) Reset() {
	fq.pos = 0
}

func newFifoQueue() *FifoQueue {
	tmp := new(FifoQueue)
	tmp.mutex = &sync.Mutex{}
	tmp.pos = 0
	return tmp
}
