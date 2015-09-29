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

import (
	"testing"
)

func TestPQ(t *testing.T) {
	t.Parallel()

	pq := NewPriorityQueue()
	if pq.Len() != 0 {
		t.Fatalf("pq should be empty")
	}

	now := NumericPriority(0)
	now2 := NumericPriority(now + 2)
	pq.Push(NewItem(nil, now2))
	if pq.Len() != 1 {
		t.Fatalf("pq.len should be 1")
	}
	x := pq.Pop()
	if x == nil {
		t.Fatalf("x is nil")
	}
	if pq.Len() != 0 {
		t.Fatalf("pq should be empty")
	}
	item := x.(Item)
	if !item.Priority().Equal(now2) {
		t.Fatalf("item.priority != now2")
	}

	pq.Push(NewItem(nil, now2))
	pq.Push(NewItem(nil, now2))
	pq.Push(NewItem(nil, now2))
	pq.Push(NewItem(nil, now2))
	pq.Push(NewItem(nil, now2))
	pq.Pop()
	pq.Pop()
	pq.Pop()
	pq.Pop()
	pq.Pop()
	if pq.Len() != 0 {
		t.Fatalf("pq should be empty")
	}
	now4 := NumericPriority(now + 4)
	now6 := NumericPriority(now + 6)
	pq.Push(NewItem(nil, now2))
	pq.Push(NewItem(nil, now4))
	pq.Push(NewItem(nil, now6))
	pq.Swap(0, 2)

	qList := *pq
	if !qList[0].Priority().Equal(now6) || !qList[2].Priority().Equal(now2) {
		t.Fatalf("swap failed")
	}

	if pq.Less(1, 2) {
		t.Fatalf("now4 < now2")
	}
}

func TestPopEmptyPQ(t *testing.T) {
	t.Parallel()
	defer func() {
		if r := recover(); r == nil {
			t.Fatalf("Expected panic from popping an empty PQ")
		}
	}()
	var pq Queue
	pq.Pop()
}
