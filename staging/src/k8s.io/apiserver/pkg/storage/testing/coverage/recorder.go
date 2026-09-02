/*
Copyright The Kubernetes Authors.

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

package coverage

import "sync"

// Recorder counts how many times each State has been observed by a Wrap-ed
// storage.Interface. It is safe for concurrent use.
type Recorder struct {
	mu   sync.Mutex
	seen map[State]int
}

// NewRecorder returns an empty Recorder.
func NewRecorder() *Recorder {
	return &Recorder{seen: make(map[State]int)}
}

// Observe records one occurrence of state.
func (r *Recorder) Observe(state State) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.seen[state]++
}

// Counts returns a snapshot of how many times each observed State occurred.
// States never observed are absent from the result.
func (r *Recorder) Counts() map[State]int {
	r.mu.Lock()
	defer r.mu.Unlock()
	out := make(map[State]int, len(r.seen))
	for k, v := range r.seen {
		out[k] = v
	}
	return out
}
