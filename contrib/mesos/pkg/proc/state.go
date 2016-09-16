/*
Copyright 2015 The Kubernetes Authors.

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

package proc

import (
	"sync/atomic"
)

type stateType int32

const (
	stateNew stateType = iota
	stateRunning
	stateTerminal
)

func (s *stateType) get() stateType {
	return stateType(atomic.LoadInt32((*int32)(s)))
}

func (s *stateType) transition(from, to stateType) bool {
	return atomic.CompareAndSwapInt32((*int32)(s), int32(from), int32(to))
}

func (s *stateType) transitionTo(to stateType, unless ...stateType) bool {
	if len(unless) == 0 {
		atomic.StoreInt32((*int32)(s), int32(to))
		return true
	}
	for {
		state := s.get()
		for _, x := range unless {
			if state == x {
				return false
			}
		}
		if s.transition(state, to) {
			return true
		}
	}
}
