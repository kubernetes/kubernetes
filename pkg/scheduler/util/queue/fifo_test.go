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

	"github.com/stretchr/testify/require"
)

func verifyPop(t *testing.T, expectedValue int, expectedOk bool, queue *FIFO[int]) {
	t.Helper()
	actual, ok := queue.Pop()
	require.Equal(t, expectedOk, ok)
	require.Equal(t, expectedValue, actual)
}

func verifyEmpty(t *testing.T, queue *FIFO[int]) {
	t.Helper()
	require.Equal(t, 0, queue.Len())
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
	require.Equal(t, 1, queue.Len())
	verifyPop(t, expected, true, &queue)
	verifyEmpty(t, &queue)
}

// Pushes some elements, pops all of them, then the same again.
func TestWrapAroundEmpty(t *testing.T) {
	var queue FIFO[int]

	for i := 0; i < 5; i++ {
		queue.Push(i)
	}
	require.Equal(t, 5, queue.Len())
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
	require.Equal(t, 5, queue.Len())
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
	require.Equal(t, normalSize*2, queue.Len())
	require.LessOrEqual(t, 2*normalSize, len(queue.elements))

	// Pop all, should be shrunken when done.
	for i := 0; i < normalSize*2; i++ {
		verifyPop(t, i, true, &queue)
	}
	require.Equal(t, 0, queue.Len())
	require.Equal(t, normalSize, len(queue.elements))

	// Still usable after shrinking?
	queue.Push(42)
	verifyPop(t, 42, true, &queue)
	require.Equal(t, 0, queue.Len())
	require.Equal(t, normalSize, len(queue.elements))
}
