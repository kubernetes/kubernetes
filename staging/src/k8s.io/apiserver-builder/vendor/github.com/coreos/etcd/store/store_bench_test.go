// Copyright 2015 The etcd Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package store

import (
	"encoding/json"
	"fmt"
	"runtime"
	"testing"
)

func BenchmarkStoreSet128Bytes(b *testing.B) {
	benchStoreSet(b, 128, nil)
}

func BenchmarkStoreSet1024Bytes(b *testing.B) {
	benchStoreSet(b, 1024, nil)
}

func BenchmarkStoreSet4096Bytes(b *testing.B) {
	benchStoreSet(b, 4096, nil)
}

func BenchmarkStoreSetWithJson128Bytes(b *testing.B) {
	benchStoreSet(b, 128, json.Marshal)
}

func BenchmarkStoreSetWithJson1024Bytes(b *testing.B) {
	benchStoreSet(b, 1024, json.Marshal)
}

func BenchmarkStoreSetWithJson4096Bytes(b *testing.B) {
	benchStoreSet(b, 4096, json.Marshal)
}

func BenchmarkStoreDelete(b *testing.B) {
	b.StopTimer()

	s := newStore()
	kvs, _ := generateNRandomKV(b.N, 128)

	memStats := new(runtime.MemStats)
	runtime.GC()
	runtime.ReadMemStats(memStats)

	for i := 0; i < b.N; i++ {
		_, err := s.Set(kvs[i][0], false, kvs[i][1], TTLOptionSet{ExpireTime: Permanent})
		if err != nil {
			panic(err)
		}
	}

	setMemStats := new(runtime.MemStats)
	runtime.GC()
	runtime.ReadMemStats(setMemStats)

	b.StartTimer()

	for i := range kvs {
		s.Delete(kvs[i][0], false, false)
	}

	b.StopTimer()

	// clean up
	e, err := s.Get("/", false, false)
	if err != nil {
		panic(err)
	}

	for _, n := range e.Node.Nodes {
		_, err := s.Delete(n.Key, true, true)
		if err != nil {
			panic(err)
		}
	}
	s.WatcherHub.EventHistory = nil

	deleteMemStats := new(runtime.MemStats)
	runtime.GC()
	runtime.ReadMemStats(deleteMemStats)

	fmt.Printf("\nBefore set Alloc: %v; After set Alloc: %v, After delete Alloc: %v\n",
		memStats.Alloc/1000, setMemStats.Alloc/1000, deleteMemStats.Alloc/1000)
}

func BenchmarkWatch(b *testing.B) {
	b.StopTimer()
	s := newStore()
	kvs, _ := generateNRandomKV(b.N, 128)
	b.StartTimer()

	memStats := new(runtime.MemStats)
	runtime.GC()
	runtime.ReadMemStats(memStats)

	for i := 0; i < b.N; i++ {
		w, _ := s.Watch(kvs[i][0], false, false, 0)

		e := newEvent("set", kvs[i][0], uint64(i+1), uint64(i+1))
		s.WatcherHub.notify(e)
		<-w.EventChan()
		s.CurrentIndex++
	}

	s.WatcherHub.EventHistory = nil
	afterMemStats := new(runtime.MemStats)
	runtime.GC()
	runtime.ReadMemStats(afterMemStats)
	fmt.Printf("\nBefore Alloc: %v; After Alloc: %v\n",
		memStats.Alloc/1000, afterMemStats.Alloc/1000)
}

func BenchmarkWatchWithSet(b *testing.B) {
	b.StopTimer()
	s := newStore()
	kvs, _ := generateNRandomKV(b.N, 128)
	b.StartTimer()

	for i := 0; i < b.N; i++ {
		w, _ := s.Watch(kvs[i][0], false, false, 0)

		s.Set(kvs[i][0], false, "test", TTLOptionSet{ExpireTime: Permanent})
		<-w.EventChan()
	}
}

func BenchmarkWatchWithSetBatch(b *testing.B) {
	b.StopTimer()
	s := newStore()
	kvs, _ := generateNRandomKV(b.N, 128)
	b.StartTimer()

	watchers := make([]Watcher, b.N)

	for i := 0; i < b.N; i++ {
		watchers[i], _ = s.Watch(kvs[i][0], false, false, 0)
	}

	for i := 0; i < b.N; i++ {
		s.Set(kvs[i][0], false, "test", TTLOptionSet{ExpireTime: Permanent})
	}

	for i := 0; i < b.N; i++ {
		<-watchers[i].EventChan()
	}

}

func BenchmarkWatchOneKey(b *testing.B) {
	s := newStore()
	watchers := make([]Watcher, b.N)

	for i := 0; i < b.N; i++ {
		watchers[i], _ = s.Watch("/foo", false, false, 0)
	}

	s.Set("/foo", false, "", TTLOptionSet{ExpireTime: Permanent})

	for i := 0; i < b.N; i++ {
		<-watchers[i].EventChan()
	}
}

func benchStoreSet(b *testing.B, valueSize int, process func(interface{}) ([]byte, error)) {
	s := newStore()
	b.StopTimer()
	kvs, size := generateNRandomKV(b.N, valueSize)
	b.StartTimer()

	for i := 0; i < b.N; i++ {
		resp, err := s.Set(kvs[i][0], false, kvs[i][1], TTLOptionSet{ExpireTime: Permanent})
		if err != nil {
			panic(err)
		}

		if process != nil {
			_, err = process(resp)
			if err != nil {
				panic(err)
			}
		}
	}

	kvs = nil
	b.StopTimer()
	memStats := new(runtime.MemStats)
	runtime.GC()
	runtime.ReadMemStats(memStats)
	fmt.Printf("\nAlloc: %vKB; Data: %vKB; Kvs: %v; Alloc/Data:%v\n",
		memStats.Alloc/1000, size/1000, b.N, memStats.Alloc/size)
}

func generateNRandomKV(n int, valueSize int) ([][]string, uint64) {
	var size uint64
	kvs := make([][]string, n)
	bytes := make([]byte, valueSize)

	for i := 0; i < n; i++ {
		kvs[i] = make([]string, 2)
		kvs[i][0] = fmt.Sprintf("/%010d/%010d/%010d", n, n, n)
		kvs[i][1] = string(bytes)
		size = size + uint64(len(kvs[i][0])) + uint64(len(kvs[i][1]))
	}

	return kvs, size
}
