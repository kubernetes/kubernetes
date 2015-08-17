/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package nodecontroller

import (
	"testing"

	"k8s.io/kubernetes/pkg/util"
)

func CheckQueueEq(lhs, rhs []string) bool {
	for i := 0; i < len(lhs); i++ {
		if rhs[i] != lhs[i] {
			return false
		}
	}
	return true
}

func CheckSetEq(lhs, rhs util.StringSet) bool {
	return lhs.HasAll(rhs.List()...) && rhs.HasAll(lhs.List()...)
}

func TestAddNode(t *testing.T) {
	evictor := NewPodEvictor(util.NewFakeRateLimiter())
	evictor.AddNodeToEvict("first")
	evictor.AddNodeToEvict("second")
	evictor.AddNodeToEvict("third")

	queuePattern := []string{"first", "second", "third"}
	if len(evictor.queue.queue) != len(queuePattern) {
		t.Fatalf("Queue %v should have lenght %d", evictor.queue.queue, len(queuePattern))
	}
	if !CheckQueueEq(queuePattern, evictor.queue.queue) {
		t.Errorf("Invalid queue. Got %v, expected %v", evictor.queue.queue, queuePattern)
	}

	setPattern := util.NewStringSet("first", "second", "third")
	if len(evictor.queue.set) != len(setPattern) {
		t.Fatalf("Map %v should have length %d", evictor.queue.set, len(setPattern))
	}
	if !CheckSetEq(setPattern, evictor.queue.set) {
		t.Errorf("Invalid map. Got %v, expected %v", evictor.queue.set, setPattern)
	}
}

func TestDelNode(t *testing.T) {
	evictor := NewPodEvictor(util.NewFakeRateLimiter())
	evictor.AddNodeToEvict("first")
	evictor.AddNodeToEvict("second")
	evictor.AddNodeToEvict("third")
	evictor.RemoveNodeToEvict("first")

	queuePattern := []string{"second", "third"}
	if len(evictor.queue.queue) != len(queuePattern) {
		t.Fatalf("Queue %v should have length %d", evictor.queue.queue, len(queuePattern))
	}
	if !CheckQueueEq(queuePattern, evictor.queue.queue) {
		t.Errorf("Invalid queue. Got %v, expected %v", evictor.queue.queue, queuePattern)
	}

	setPattern := util.NewStringSet("second", "third")
	if len(evictor.queue.set) != len(setPattern) {
		t.Fatalf("Map %v should have length %d", evictor.queue.set, len(setPattern))
	}
	if !CheckSetEq(setPattern, evictor.queue.set) {
		t.Errorf("Invalid map. Got %v, expected %v", evictor.queue.set, setPattern)
	}

	evictor = NewPodEvictor(util.NewFakeRateLimiter())
	evictor.AddNodeToEvict("first")
	evictor.AddNodeToEvict("second")
	evictor.AddNodeToEvict("third")
	evictor.RemoveNodeToEvict("second")

	queuePattern = []string{"first", "third"}
	if len(evictor.queue.queue) != len(queuePattern) {
		t.Fatalf("Queue %v should have lenght %d", evictor.queue.queue, len(queuePattern))
	}
	if !CheckQueueEq(queuePattern, evictor.queue.queue) {
		t.Errorf("Invalid queue. Got %v, expected %v", evictor.queue.queue, queuePattern)
	}

	setPattern = util.NewStringSet("first", "third")
	if len(evictor.queue.set) != len(setPattern) {
		t.Fatalf("Map %v should have length %d", evictor.queue.set, len(setPattern))
	}
	if !CheckSetEq(setPattern, evictor.queue.set) {
		t.Errorf("Invalid map. Got %v, expected %v", evictor.queue.set, setPattern)
	}

	evictor = NewPodEvictor(util.NewFakeRateLimiter())
	evictor.AddNodeToEvict("first")
	evictor.AddNodeToEvict("second")
	evictor.AddNodeToEvict("third")
	evictor.RemoveNodeToEvict("third")

	queuePattern = []string{"first", "second"}
	if len(evictor.queue.queue) != len(queuePattern) {
		t.Fatalf("Queue %v should have lenght %d", evictor.queue.queue, len(queuePattern))
	}
	if !CheckQueueEq(queuePattern, evictor.queue.queue) {
		t.Errorf("Invalid queue. Got %v, expected %v", evictor.queue.queue, queuePattern)
	}

	setPattern = util.NewStringSet("first", "second")
	if len(evictor.queue.set) != len(setPattern) {
		t.Fatalf("Map %v should have length %d", evictor.queue.set, len(setPattern))
	}
	if !CheckSetEq(setPattern, evictor.queue.set) {
		t.Errorf("Invalid map. Got %v, expected %v", evictor.queue.set, setPattern)
	}
}

func TestEvictNode(t *testing.T) {
	evictor := NewPodEvictor(util.NewFakeRateLimiter())
	evictor.AddNodeToEvict("first")
	evictor.AddNodeToEvict("second")
	evictor.AddNodeToEvict("third")
	evictor.RemoveNodeToEvict("second")

	deletedMap := util.NewStringSet()
	evictor.TryEvict(func(nodeName string) { deletedMap.Insert(nodeName) })

	setPattern := util.NewStringSet("first", "third")
	if len(deletedMap) != len(setPattern) {
		t.Fatalf("Map %v should have length %d", evictor.queue.set, len(setPattern))
	}
	if !CheckSetEq(setPattern, deletedMap) {
		t.Errorf("Invalid map. Got %v, expected %v", deletedMap, setPattern)
	}
}
