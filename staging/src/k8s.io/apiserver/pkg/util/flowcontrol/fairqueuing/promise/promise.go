/*
Copyright 2019 The Kubernetes Authors.

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

package promise

import (
	"sync"
)

// promise implements the WriteOnce interface.
type promise struct {
	doneCh  <-chan struct{}
	doneVal interface{}
	setCh   chan struct{}
	onceler sync.Once
	value   interface{}
}

var _ WriteOnce = &promise{}

// NewWriteOnce makes a new thread-safe WriteOnce.
//
// If `initial` is non-nil then that value is Set at creation time.
//
// If a `Get` is waiting soon after `doneCh` becomes selectable (which
// never happens for the nil channel) then `Set(doneVal)` effectively
// happens at that time.
func NewWriteOnce(initial interface{}, doneCh <-chan struct{}, doneVal interface{}) WriteOnce {
	p := &promise{
		doneCh:  doneCh,
		doneVal: doneVal,
		setCh:   make(chan struct{}),
	}
	if initial != nil {
		p.Set(initial)
	}
	return p
}

func (p *promise) Get() interface{} {
	select {
	case <-p.setCh:
	case <-p.doneCh:
		p.Set(p.doneVal)
	}
	return p.value
}

func (p *promise) Set(value interface{}) bool {
	var ans bool
	p.onceler.Do(func() {
		p.value = value
		close(p.setCh)
		ans = true
	})
	return ans
}
