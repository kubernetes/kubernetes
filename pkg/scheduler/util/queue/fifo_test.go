/*
Copyright 2024 The Kubernetes Authors.

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

package queue

import (
	"testing"
)

func verifyPop(t *testing.T, expectedValue int, expectedOk bool, queue *FIFO[int]) {
	t.Helper()
	actual, ok := queue.Pop()
	if ok != expectedOk {
		t.Errorf("Expected queue.Pop() status: %t, actual status: %t", expectedOk, ok)
	}
	if actual != expectedValue {
		t.Errorf("Expected queue.Pop() value: %d, actual status: %d", expectedValue, actual)
	}
}

func verifyEmpty(t *testing.T, queue *FIFO[int]) {
	t.Helper()
	if queue.Len() != 0 {
		t.Errorf("Expected empty queue, actual Len: %d", queue.Len())
	}
	verifyPop(t, 0, false, queue)
}

func TestNull(t *testing.T) {
	var queue FIFO[int]
	verifyEmpty(t, &queue)
}

func TestOnePushPop(t *testing.T) {
	var queue FIFO[int]

	expected := 10
	queue.Push(10)
	if queue.Len() != 1 {
		t.Errorf("Expected queue length as 1, actual len: %d", queue.Len())
	}
	verifyPop(t, expected, true, &queue)
	verifyEmpty(t, &queue)
}

// Pushes some elements, pops all of them, then the same again.
func TestWrapAroundEmpty(t *testing.T) {
	var queue FIFO[int]

	for i := 0; i < 5; i++ {
		queue.Push(i)
	}
	if queue.Len() != 5 {
		t.Errorf("Expected queue len as 5, actual len: %d", queue.Len())
	}
	for i := 0; i < 5; i++ {
		verifyPop(t, i, true, &queue)
	}
	verifyEmpty(t, &queue)

	for i := 5; i < 10; i++ {
		queue.Push(i)
	}
	for i := 5; i < 10; i++ {
		verifyPop(t, i, true, &queue)
	}
	verifyEmpty(t, &queue)
}

// Pushes some elements, pops one, adds more, then pops all.
func TestWrapAroundPartial(t *testing.T) {
	var queue FIFO[int]

	for i := 0; i < 5; i++ {
		queue.Push(i)
	}
	if queue.Len() != 5 {
		t.Errorf("Expected queue len as 5, actual len: %d", queue.Len())
	}
	verifyPop(t, 0, true, &queue)

	for i := 5; i < 10; i++ {
		queue.Push(i)
	}
	for i := 1; i < 10; i++ {
		verifyPop(t, i, true, &queue)
	}
	verifyEmpty(t, &queue)
}

// Push an unusual amount of elements, pop all, and verify that
// the FIFO shrinks back again.
func TestShrink(t *testing.T) {
	var queue FIFO[int]

	for i := 0; i < normalSize*2; i++ {
		queue.Push(i)
	}
	if queue.Len() != normalSize*2 {
		t.Errorf("Expected queue len as %d, actual len: %d", normalSize*2, queue.Len())
	}
	if len(queue.elements) < 2*normalSize {
		t.Errorf("Expected queue elements len as %d, actual len: %d", 2*normalSize, len(queue.elements))
	}

	// Pop all, should be shrunken when done.
	for i := 0; i < normalSize*2; i++ {
		verifyPop(t, i, true, &queue)
	}
	if queue.Len() != 0 {
		t.Errorf("Expected queue len as 0, actual len: %d", queue.Len())
	}
	if len(queue.elements) != normalSize {
		t.Errorf("Expected queue elements len as %d, actual len: %d", normalSize, len(queue.elements))
	}

	// Still usable after shrinking?
	queue.Push(42)
	verifyPop(t, 42, true, &queue)
	if queue.Len() != 0 {
		t.Errorf("Expected queue len as 0, actual len: %d", queue.Len())
	}
	if len(queue.elements) != normalSize {
		t.Errorf("Expected queue elements len as %d, actual len: %d", normalSize, len(queue.elements))
	}
}
