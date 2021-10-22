/*
 *
 * Copyright 2017 gRPC authors.
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

// Package primitives_test contains benchmarks for various synchronization primitives
// available in Go.
package primitives_test

import (
	"fmt"
	"sync"
	"sync/atomic"
	"testing"
	"time"
	"unsafe"
)

func BenchmarkSelectClosed(b *testing.B) {
	c := make(chan struct{})
	close(c)
	x := 0
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		select {
		case <-c:
			x++
		default:
		}
	}
	b.StopTimer()
	if x != b.N {
		b.Fatal("error")
	}
}

func BenchmarkSelectOpen(b *testing.B) {
	c := make(chan struct{})
	x := 0
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		select {
		case <-c:
		default:
			x++
		}
	}
	b.StopTimer()
	if x != b.N {
		b.Fatal("error")
	}
}

func BenchmarkAtomicBool(b *testing.B) {
	c := int32(0)
	x := 0
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if atomic.LoadInt32(&c) == 0 {
			x++
		}
	}
	b.StopTimer()
	if x != b.N {
		b.Fatal("error")
	}
}

func BenchmarkAtomicValueLoad(b *testing.B) {
	c := atomic.Value{}
	c.Store(0)
	x := 0
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if c.Load().(int) == 0 {
			x++
		}
	}
	b.StopTimer()
	if x != b.N {
		b.Fatal("error")
	}
}

func BenchmarkAtomicValueStore(b *testing.B) {
	c := atomic.Value{}
	v := 123
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		c.Store(v)
	}
	b.StopTimer()
}

func BenchmarkMutex(b *testing.B) {
	c := sync.Mutex{}
	x := 0
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		c.Lock()
		x++
		c.Unlock()
	}
	b.StopTimer()
	if x != b.N {
		b.Fatal("error")
	}
}

func BenchmarkRWMutex(b *testing.B) {
	c := sync.RWMutex{}
	x := 0
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		c.RLock()
		x++
		c.RUnlock()
	}
	b.StopTimer()
	if x != b.N {
		b.Fatal("error")
	}
}

func BenchmarkRWMutexW(b *testing.B) {
	c := sync.RWMutex{}
	x := 0
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		c.Lock()
		x++
		c.Unlock()
	}
	b.StopTimer()
	if x != b.N {
		b.Fatal("error")
	}
}

func BenchmarkMutexWithDefer(b *testing.B) {
	c := sync.Mutex{}
	x := 0
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		func() {
			c.Lock()
			defer c.Unlock()
			x++
		}()
	}
	b.StopTimer()
	if x != b.N {
		b.Fatal("error")
	}
}

func BenchmarkMutexWithClosureDefer(b *testing.B) {
	c := sync.Mutex{}
	x := 0
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		func() {
			c.Lock()
			defer func() { c.Unlock() }()
			x++
		}()
	}
	b.StopTimer()
	if x != b.N {
		b.Fatal("error")
	}
}

func BenchmarkMutexWithoutDefer(b *testing.B) {
	c := sync.Mutex{}
	x := 0
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		func() {
			c.Lock()
			x++
			c.Unlock()
		}()
	}
	b.StopTimer()
	if x != b.N {
		b.Fatal("error")
	}
}

func BenchmarkAtomicAddInt64(b *testing.B) {
	var c int64
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		atomic.AddInt64(&c, 1)
	}
	b.StopTimer()
	if c != int64(b.N) {
		b.Fatal("error")
	}
}

func BenchmarkAtomicTimeValueStore(b *testing.B) {
	var c atomic.Value
	t := time.Now()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		c.Store(t)
	}
	b.StopTimer()
}

func BenchmarkAtomic16BValueStore(b *testing.B) {
	var c atomic.Value
	t := struct {
		a int64
		b int64
	}{
		123, 123,
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		c.Store(t)
	}
	b.StopTimer()
}

func BenchmarkAtomic32BValueStore(b *testing.B) {
	var c atomic.Value
	t := struct {
		a int64
		b int64
		c int64
		d int64
	}{
		123, 123, 123, 123,
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		c.Store(t)
	}
	b.StopTimer()
}

func BenchmarkAtomicPointerStore(b *testing.B) {
	t := 123
	var up unsafe.Pointer
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		atomic.StorePointer(&up, unsafe.Pointer(&t))
	}
	b.StopTimer()
}

func BenchmarkAtomicTimePointerStore(b *testing.B) {
	t := time.Now()
	var up unsafe.Pointer
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		atomic.StorePointer(&up, unsafe.Pointer(&t))
	}
	b.StopTimer()
}

func BenchmarkStoreContentionWithAtomic(b *testing.B) {
	t := 123
	var c unsafe.Pointer
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			atomic.StorePointer(&c, unsafe.Pointer(&t))
		}
	})
}

func BenchmarkStoreContentionWithMutex(b *testing.B) {
	t := 123
	var mu sync.Mutex
	var c int

	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			mu.Lock()
			c = t
			mu.Unlock()
		}
	})
	_ = c
}

type dummyStruct struct {
	a int64
	b time.Time
}

func BenchmarkStructStoreContention(b *testing.B) {
	d := dummyStruct{}
	dp := unsafe.Pointer(&d)
	t := time.Now()
	for _, j := range []int{100000000, 10000, 0} {
		for _, i := range []int{100000, 10} {
			b.Run(fmt.Sprintf("CAS/%v/%v", j, i), func(b *testing.B) {
				b.SetParallelism(i)
				b.RunParallel(func(pb *testing.PB) {
					n := &dummyStruct{
						b: t,
					}
					for pb.Next() {
						for y := 0; y < j; y++ {
						}
						for {
							v := (*dummyStruct)(atomic.LoadPointer(&dp))
							n.a = v.a + 1
							if atomic.CompareAndSwapPointer(&dp, unsafe.Pointer(v), unsafe.Pointer(n)) {
								n = v
								break
							}
						}
					}
				})
			})
		}
	}

	var mu sync.Mutex
	for _, j := range []int{100000000, 10000, 0} {
		for _, i := range []int{100000, 10} {
			b.Run(fmt.Sprintf("Mutex/%v/%v", j, i), func(b *testing.B) {
				b.SetParallelism(i)
				b.RunParallel(func(pb *testing.PB) {
					for pb.Next() {
						for y := 0; y < j; y++ {
						}
						mu.Lock()
						d.a++
						d.b = t
						mu.Unlock()
					}
				})
			})
		}
	}
}

type myFooer struct{}

func (myFooer) Foo() {}

type fooer interface {
	Foo()
}

func BenchmarkInterfaceTypeAssertion(b *testing.B) {
	// Call a separate function to avoid compiler optimizations.
	runInterfaceTypeAssertion(b, myFooer{})
}

func runInterfaceTypeAssertion(b *testing.B, fer interface{}) {
	x := 0
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, ok := fer.(fooer); ok {
			x++
		}
	}
	b.StopTimer()
	if x != b.N {
		b.Fatal("error")
	}
}

func BenchmarkStructTypeAssertion(b *testing.B) {
	// Call a separate function to avoid compiler optimizations.
	runStructTypeAssertion(b, myFooer{})
}

func runStructTypeAssertion(b *testing.B, fer interface{}) {
	x := 0
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, ok := fer.(myFooer); ok {
			x++
		}
	}
	b.StopTimer()
	if x != b.N {
		b.Fatal("error")
	}
}
