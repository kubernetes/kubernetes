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

package results

import (
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"k8s.io/kubernetes/pkg/api"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/util"
)

func TestCacheOperations(t *testing.T) {
	m := NewManager()

	unsetID := kubecontainer.ContainerID{"test", "unset"}
	setID := kubecontainer.ContainerID{"test", "set"}

	_, found := m.Get(unsetID)
	assert.False(t, found, "unset result found")

	m.Set(setID, Success, nil)
	result, found := m.Get(setID)
	assert.True(t, result == Success, "set result")
	assert.True(t, found, "set result found")

	m.Remove(setID)
	_, found = m.Get(setID)
	assert.False(t, found, "removed result found")
}

func TestUpdates(t *testing.T) {
	m := NewManagerWithUpdates()

	pod := &api.Pod{ObjectMeta: api.ObjectMeta{Name: "test-pod"}}
	fooID := kubecontainer.ContainerID{"test", "foo"}
	barID := kubecontainer.ContainerID{"test", "bar"}

	expectUpdate := func(expected Update, msg string) {
		select {
		case u := <-m.Updates():
			if expected != u {
				t.Errorf("Expected update %v, recieved %v: %s %s", expected, u, msg)
			}
		case <-time.After(util.ForeverTestTimeout):
			t.Errorf("Timed out waiting for update %v: %s", expected, msg)
		}
	}

	expectNoUpdate := func(msg string) {
		// NOTE: Since updates are accumulated asynchronously, this method is not guaranteed to fail
		// when it should. In the event it misses a failure, the following calls to expectUpdate should
		// still fail.
		select {
		case u := <-m.Updates():
			t.Errorf("Unexpected update %v: %s", u, msg)
		default:
			// Pass
		}
	}

	// New result should always push an update.
	m.Set(fooID, Success, pod)
	expectUpdate(Update{fooID, Success, pod}, "new success")

	m.Set(barID, Failure, pod)
	expectUpdate(Update{barID, Failure, pod}, "new failure")

	// Unchanged results should not send an update.
	m.Set(fooID, Success, pod)
	expectNoUpdate("unchanged foo")

	m.Set(barID, Failure, pod)
	expectNoUpdate("unchanged bar")

	// Changed results should send an update.
	m.Set(fooID, Failure, pod)
	expectUpdate(Update{fooID, Failure, pod}, "changed foo")

	m.Set(barID, Success, pod)
	expectUpdate(Update{barID, Success, pod}, "changed bar")
}
