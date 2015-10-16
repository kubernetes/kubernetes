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
)

type Wait interface {
	Register(id uint64) <-chan interface{}
	Trigger(id uint64, x interface{})
}

type List struct {
	l sync.Mutex
	m map[uint64]chan interface{}
}

func New() *List {
	return &List{m: make(map[uint64]chan interface{})}
}

func (w *List) Register(id uint64) <-chan interface{} {
	w.l.Lock()
	defer w.l.Unlock()
	ch := w.m[id]
	if ch == nil {
		ch = make(chan interface{}, 1)
		w.m[id] = ch
	}
	return ch
}

func (w *List) Trigger(id uint64, x interface{}) {
	w.l.Lock()
	ch := w.m[id]
	delete(w.m, id)
	w.l.Unlock()
	if ch != nil {
		ch <- x
		close(ch)
	}
}
