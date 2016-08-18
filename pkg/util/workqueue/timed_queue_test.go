/*
Copyright 2016 The Kubernetes Authors.

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

package workqueue

import (
	"testing"

	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/util/clock"
)

func TestNoMemoryLeak(t *testing.T) {
	timedQueue := NewTimedWorkQueue(clock.RealClock{})
	pod := &v1.Pod{}
	timedQueue.Add(pod)
	item, _, _ := timedQueue.Get()
	timedQueue.AddWithTimestamp(item)
	// The item should still be in the timedQueue.
	timedQueue.Done(item)
	item, _, _ = timedQueue.Get()
	timedQueue.Done(item)
	if len(timedQueue.Type.processing) != 0 {
		t.Errorf("expect timedQueue.Type.processing to be empty!")
	}
}
