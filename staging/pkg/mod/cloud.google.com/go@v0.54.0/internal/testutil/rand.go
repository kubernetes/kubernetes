// Copyright 2018 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package testutil

import (
	"math/rand"
	"sync"
	"time"
)

// NewRand creates a new *rand.Rand seeded with t. The return value is safe for use
// with multiple goroutines.
func NewRand(t time.Time) *rand.Rand {
	s := &lockedSource{src: rand.NewSource(t.UnixNano())}
	return rand.New(s)
}

// lockedSource makes a rand.Source safe for use by multiple goroutines.
type lockedSource struct {
	mu  sync.Mutex
	src rand.Source
}

func (ls *lockedSource) Int63() int64 {
	ls.mu.Lock()
	defer ls.mu.Unlock()
	return ls.src.Int63()
}

func (ls *lockedSource) Seed(int64) {
	panic("shouldn't be calling Seed")
}
