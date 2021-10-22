// +build !appengine

/*
 *
 * Copyright 2019 gRPC authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

package buffer

import (
	"fmt"
	"sync"
	"testing"
	"time"
)

func TestCircularBufferSerial(t *testing.T) {
	var size, i uint32
	var result []interface{}

	size = 1 << 15
	cb, err := NewCircularBuffer(size)
	if err != nil {
		t.Fatalf("error allocating CircularBuffer: %v", err)
	}

	for i = 0; i < size/2; i++ {
		cb.Push(i)
	}

	result = cb.Drain()
	if uint32(len(result)) != size/2 {
		t.Fatalf("len(result) = %d; want %d", len(result), size/2)
	}

	// The returned result isn't necessarily sorted.
	seen := make(map[uint32]bool)
	for _, r := range result {
		seen[r.(uint32)] = true
	}

	for i = 0; i < uint32(len(result)); i++ {
		if !seen[i] {
			t.Fatalf("seen[%d] = false; want true", i)
		}
	}

	for i = 0; i < size; i++ {
		cb.Push(i)
	}

	result = cb.Drain()
	if uint32(len(result)) != size {
		t.Fatalf("len(result) = %d; want %d", len(result), size/2)
	}
}

func TestCircularBufferOverflow(t *testing.T) {
	var size, i uint32
	var result []interface{}

	size = 1 << 10
	cb, err := NewCircularBuffer(size)
	if err != nil {
		t.Fatalf("error allocating CircularBuffer: %v", err)
	}

	for i = 0; i < 10*size; i++ {
		cb.Push(i)
	}

	result = cb.Drain()

	if uint32(len(result)) != size {
		t.Fatalf("len(result) = %d; want %d", len(result), size)
	}

	for idx, x := range result {
		if x.(uint32) < size {
			t.Fatalf("result[%d] = %d; want it to be >= %d", idx, x, size)
		}
	}
}

func TestCircularBufferConcurrent(t *testing.T) {
	for tn := 0; tn < 2; tn++ {
		var size uint32
		var result []interface{}

		size = 1 << 6
		cb, err := NewCircularBuffer(size)
		if err != nil {
			t.Fatalf("error allocating CircularBuffer: %v", err)
		}

		type item struct {
			R uint32
			N uint32
			T time.Time
		}

		var wg sync.WaitGroup
		for r := uint32(0); r < 1024; r++ {
			wg.Add(1)
			go func(r uint32) {
				for n := uint32(0); n < size; n++ {
					cb.Push(item{R: r, N: n, T: time.Now()})
				}
				wg.Done()
			}(r)
		}

		// Wait for all goroutines to finish only in one test. Draining
		// concurrently while Pushes are still happening will test for races in the
		// Draining lock.
		if tn == 0 {
			wg.Wait()
		}

		result = cb.Drain()

		// Can't expect the buffer to be full if the Pushes aren't necessarily done.
		if tn == 0 {
			if uint32(len(result)) != size {
				t.Fatalf("len(result) = %d; want %d", len(result), size)
			}
		}

		// There can be absolutely no expectation on the order of the data returned
		// by Drain because: (a) everything is happening concurrently (b) a
		// round-robin is used to write to different queues (and therefore
		// different cachelines) for less write contention.

		// Wait for all goroutines to complete before moving on to other tests. If
		// the benchmarks run after this, it might affect performance unfairly.
		wg.Wait()
	}
}

func BenchmarkCircularBuffer(b *testing.B) {
	x := 1
	for size := 1 << 16; size <= 1<<20; size <<= 1 {
		for routines := 1; routines <= 1<<8; routines <<= 1 {
			b.Run(fmt.Sprintf("goroutines:%d/size:%d", routines, size), func(b *testing.B) {
				cb, err := NewCircularBuffer(uint32(size))
				if err != nil {
					b.Fatalf("error allocating CircularBuffer: %v", err)
				}

				perRoutine := b.N / routines
				var wg sync.WaitGroup
				for r := 0; r < routines; r++ {
					wg.Add(1)
					go func() {
						for i := 0; i < perRoutine; i++ {
							cb.Push(&x)
						}
						wg.Done()
					}()
				}
				wg.Wait()
			})
		}
	}
}
