// Copyright 2015 CoreOS, Inc.
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

// Package idutil implements utility functions for generating unique,
// randomized ids.
package idutil

import (
	"math"
	"sync"
	"time"
)

const (
	tsLen     = 5 * 8
	cntLen    = 8
	suffixLen = tsLen + cntLen
)

// Generator generates unique identifiers based on counters, timestamps, and
// a node member ID.
//
// The initial id is in this format:
// High order byte is memberID, next 5 bytes are from timestamp,
// and low order 2 bytes are 0s.
// | prefix   | suffix              |
// | 2 bytes  | 5 bytes   | 1 byte  |
// | memberID | timestamp | cnt     |
//
// The timestamp 5 bytes is different when the machine is restart
// after 1 ms and before 35 years.
//
// It increases suffix to generate the next id.
// The count field may overflow to timestamp field, which is intentional.
// It helps to extend the event window to 2^56. This doesn't break that
// id generated after restart is unique because etcd throughput is <<
// 256req/ms(250k reqs/second).
type Generator struct {
	mu sync.Mutex
	// high order 2 bytes
	prefix uint64
	// low order 6 bytes
	suffix uint64
}

func NewGenerator(memberID uint16, now time.Time) *Generator {
	prefix := uint64(memberID) << suffixLen
	unixMilli := uint64(now.UnixNano()) / uint64(time.Millisecond/time.Nanosecond)
	suffix := lowbit(unixMilli, tsLen) << cntLen
	return &Generator{
		prefix: prefix,
		suffix: suffix,
	}
}

// Next generates a id that is unique.
func (g *Generator) Next() uint64 {
	g.mu.Lock()
	defer g.mu.Unlock()
	g.suffix++
	id := g.prefix | lowbit(g.suffix, suffixLen)
	return id
}

func lowbit(x uint64, n uint) uint64 {
	return x & (math.MaxUint64 >> (64 - n))
}
