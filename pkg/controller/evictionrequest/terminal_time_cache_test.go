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

package evictionrequest

import (
	"testing"
	"time"
)

func TestTerminalTimeCache(t *testing.T) {
	cache := NewTerminalTimeCache()

	now := time.Now()
	cache.recordTargetTerminalTime("foo", now)
	cache.recordTargetTerminalTime("bar", now.Add(1*time.Minute))
	cache.recordTargetTerminalTime("baz", now.Add(2*time.Minute))

	if got := cache.getTargetTerminalTime("foo"); !got.Equal(now) {
		t.Errorf("got terminal time %v, expected %v", got, now)
	}
	cache.clearTargetTerminalTime("foo")
	if got := cache.getTargetTerminalTime("foo"); got != nil {
		t.Errorf("got terminal time %v, expected nil", got)
	}
	if got := cache.getTargetTerminalTime("baz"); !got.Equal(now.Add(2 * time.Minute)) {
		t.Errorf("got terminal time %v, expected %v", got, now)
	}
}
