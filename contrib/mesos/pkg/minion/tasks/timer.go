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

package tasks

import (
	"time"
)

type timer interface {
	set(time.Duration)
	discard()
	await() <-chan time.Time
}

type realTimer struct {
	*time.Timer
}

func (t *realTimer) set(d time.Duration) {
	if t.Timer == nil {
		t.Timer = time.NewTimer(d)
	} else {
		t.Reset(d)
	}
}

func (t *realTimer) await() <-chan time.Time {
	if t.Timer == nil {
		return nil
	}
	return t.C
}

func (t *realTimer) discard() {
	if t.Timer != nil {
		t.Stop()
	}
}
