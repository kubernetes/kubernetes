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

package testutil

import (
	"net/http"
	"sync"
)

type PauseableHandler struct {
	Next   http.Handler
	mu     sync.Mutex
	paused bool
}

func (ph *PauseableHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	ph.mu.Lock()
	paused := ph.paused
	ph.mu.Unlock()
	if !paused {
		ph.Next.ServeHTTP(w, r)
	} else {
		hj, ok := w.(http.Hijacker)
		if !ok {
			panic("webserver doesn't support hijacking")
		}
		conn, _, err := hj.Hijack()
		if err != nil {
			panic(err.Error())
		}
		conn.Close()
	}
}

func (ph *PauseableHandler) Pause() {
	ph.mu.Lock()
	defer ph.mu.Unlock()
	ph.paused = true
}

func (ph *PauseableHandler) Resume() {
	ph.mu.Lock()
	defer ph.mu.Unlock()
	ph.paused = false
}
