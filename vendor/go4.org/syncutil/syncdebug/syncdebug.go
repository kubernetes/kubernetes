/*
Copyright 2013 The Camlistore Authors

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

// Package syncdebug contains facilities for debugging synchronization
// problems.
package syncdebug // import "go4.org/syncutil/syncdebug"

import (
	"bytes"
	"fmt"
	"log"
	"runtime"
	"sync"
	"sync/atomic"
	"time"

	"go4.org/strutil"
)

// RWMutexTracker is a sync.RWMutex that tracks who owns the current
// exclusive lock. It's used for debugging deadlocks.
type RWMutexTracker struct {
	mu sync.RWMutex

	// Atomic counters for number waiting and having read and write locks.
	nwaitr int32
	nwaitw int32
	nhaver int32
	nhavew int32 // should always be 0 or 1

	logOnce sync.Once

	hmu    sync.Mutex
	holder []byte
	holdr  map[int64]bool // goroutines holding read lock
}

const stackBufSize = 16 << 20

var stackBuf = make(chan []byte, 8)

func getBuf() []byte {
	select {
	case b := <-stackBuf:
		return b[:stackBufSize]
	default:
		return make([]byte, stackBufSize)
	}
}

func putBuf(b []byte) {
	select {
	case stackBuf <- b:
	default:
	}
}

var goroutineSpace = []byte("goroutine ")

// GoroutineID returns the current goroutine's ID.
// Use of this function is almost always a terrible idea.
// It is also very slow.
// GoroutineID is intended only for debugging.
// In particular, it is used by syncutil.
func GoroutineID() int64 {
	b := getBuf()
	defer putBuf(b)
	b = b[:runtime.Stack(b, false)]
	// Parse the 4707 out of "goroutine 4707 ["
	b = bytes.TrimPrefix(b, goroutineSpace)
	i := bytes.IndexByte(b, ' ')
	if i < 0 {
		panic(fmt.Sprintf("No space found in %q", b))
	}
	b = b[:i]
	n, err := strutil.ParseUintBytes(b, 10, 64)
	if err != nil {
		panic(fmt.Sprintf("Failed to parse goroutine ID out of %q: %v", b, err))
	}
	return int64(n)
}

func (m *RWMutexTracker) startLogger() {
	go func() {
		var buf bytes.Buffer
		for {
			time.Sleep(1 * time.Second)
			buf.Reset()
			m.hmu.Lock()
			for gid := range m.holdr {
				fmt.Fprintf(&buf, " [%d]", gid)
			}
			m.hmu.Unlock()
			log.Printf("Mutex %p: waitW %d haveW %d   waitR %d haveR %d %s",
				m,
				atomic.LoadInt32(&m.nwaitw),
				atomic.LoadInt32(&m.nhavew),
				atomic.LoadInt32(&m.nwaitr),
				atomic.LoadInt32(&m.nhaver), buf.Bytes())
		}
	}()
}

func (m *RWMutexTracker) Lock() {
	m.logOnce.Do(m.startLogger)
	atomic.AddInt32(&m.nwaitw, 1)
	m.mu.Lock()
	atomic.AddInt32(&m.nwaitw, -1)
	atomic.AddInt32(&m.nhavew, 1)

	m.hmu.Lock()
	defer m.hmu.Unlock()
	if len(m.holder) == 0 {
		m.holder = make([]byte, stackBufSize)
	}
	m.holder = m.holder[:runtime.Stack(m.holder[:stackBufSize], false)]
	log.Printf("Lock at %s", string(m.holder))
}

func (m *RWMutexTracker) Unlock() {
	m.hmu.Lock()
	m.holder = nil
	m.hmu.Unlock()

	atomic.AddInt32(&m.nhavew, -1)
	m.mu.Unlock()
}

func (m *RWMutexTracker) RLock() {
	m.logOnce.Do(m.startLogger)
	atomic.AddInt32(&m.nwaitr, 1)

	// Catch read-write-read lock. See if somebody (us? via
	// another goroutine?) already has a read lock, and then
	// somebody else is waiting to write, meaning our second read
	// will deadlock.
	if atomic.LoadInt32(&m.nhaver) > 0 && atomic.LoadInt32(&m.nwaitw) > 0 {
		buf := getBuf()
		buf = buf[:runtime.Stack(buf, false)]
		log.Printf("Potential R-W-R deadlock at: %s", buf)
		putBuf(buf)
	}

	m.mu.RLock()
	atomic.AddInt32(&m.nwaitr, -1)
	atomic.AddInt32(&m.nhaver, 1)

	gid := GoroutineID()
	m.hmu.Lock()
	defer m.hmu.Unlock()
	if m.holdr == nil {
		m.holdr = make(map[int64]bool)
	}
	if m.holdr[gid] {
		buf := getBuf()
		buf = buf[:runtime.Stack(buf, false)]
		log.Fatalf("Recursive call to RLock: %s", buf)
	}
	m.holdr[gid] = true
}

func stack() []byte {
	buf := make([]byte, 1024)
	return buf[:runtime.Stack(buf, false)]
}

func (m *RWMutexTracker) RUnlock() {
	atomic.AddInt32(&m.nhaver, -1)

	gid := GoroutineID()
	m.hmu.Lock()
	delete(m.holdr, gid)
	m.hmu.Unlock()

	m.mu.RUnlock()
}

// Holder returns the stack trace of the current exclusive lock holder's stack
// when it acquired the lock (with Lock). It returns the empty string if the lock
// is not currently held.
func (m *RWMutexTracker) Holder() string {
	m.hmu.Lock()
	defer m.hmu.Unlock()
	return string(m.holder)
}
