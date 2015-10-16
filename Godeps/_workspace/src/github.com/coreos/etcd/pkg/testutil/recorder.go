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

package testutil

import "sync"

type Action struct {
	Name   string
	Params []interface{}
}

type Recorder struct {
	sync.Mutex
	actions []Action
}

func (r *Recorder) Record(a Action) {
	r.Lock()
	r.actions = append(r.actions, a)
	r.Unlock()
}
func (r *Recorder) Action() []Action {
	r.Lock()
	cpy := make([]Action, len(r.actions))
	copy(cpy, r.actions)
	r.Unlock()
	return cpy
}
