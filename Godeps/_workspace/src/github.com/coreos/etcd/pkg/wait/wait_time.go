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

package wait

import (
	"sync"
	"time"
)

type WaitTime interface {
	// Wait returns a chan that waits on the given deadline.
	// The chan will be triggered when Trigger is called with a
	// deadline that is later than the one it is waiting for.
	// The given deadline MUST be unique. The deadline should be
	// retrieved by calling time.Now() in most cases.
	Wait(deadline time.Time) <-chan struct{}
	// Trigger triggers all the waiting chans with an earlier deadline.
	Trigger(deadline time.Time)
}

type timeList struct {
	l sync.Mutex
	m map[int64]chan struct{}
}

func NewTimeList() *timeList {
	return &timeList{m: make(map[int64]chan struct{})}
}

func (tl *timeList) Wait(deadline time.Time) <-chan struct{} {
	tl.l.Lock()
	defer tl.l.Unlock()
	ch := make(chan struct{}, 1)
	// The given deadline SHOULD be unique.
	tl.m[deadline.UnixNano()] = ch
	return ch
}

func (tl *timeList) Trigger(deadline time.Time) {
	tl.l.Lock()
	defer tl.l.Unlock()
	for t, ch := range tl.m {
		if t < deadline.UnixNano() {
			delete(tl.m, t)
			close(ch)
		}
	}
}
