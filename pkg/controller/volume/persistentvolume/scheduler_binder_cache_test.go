/*
Copyright 2017 The Kubernetes Authors.

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

package persistentvolume

import (
	"reflect"
	"testing"

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func TestUpdateGetBindings(t *testing.T) {
	scenarios := map[string]struct {
		updateBindings []*bindingInfo
		updatePod      string
		updateNode     string

		getBindings []*bindingInfo
		getPod      string
		getNode     string
	}{
		"no-pod": {
			getPod:  "pod1",
			getNode: "node1",
		},
		"no-node": {
			updatePod:      "pod1",
			updateNode:     "node1",
			updateBindings: []*bindingInfo{},
			getPod:         "pod1",
			getNode:        "node2",
		},
		"binding-exists": {
			updatePod:      "pod1",
			updateNode:     "node1",
			updateBindings: []*bindingInfo{{pvc: &v1.PersistentVolumeClaim{ObjectMeta: metav1.ObjectMeta{Name: "pvc1"}}}},
			getPod:         "pod1",
			getNode:        "node1",
			getBindings:    []*bindingInfo{{pvc: &v1.PersistentVolumeClaim{ObjectMeta: metav1.ObjectMeta{Name: "pvc1"}}}},
		},
	}

	for name, scenario := range scenarios {
		cache := NewPodBindingCache()

		// Perform updates
		updatePod := &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: scenario.updatePod, Namespace: "ns"}}
		cache.UpdateBindings(updatePod, scenario.updateNode, scenario.updateBindings)

		// Verify updated bindings
		bindings := cache.GetBindings(updatePod, scenario.updateNode)
		if !reflect.DeepEqual(bindings, scenario.updateBindings) {
			t.Errorf("Test %v failed: returned bindings after update different. Got %+v, expected %+v", name, bindings, scenario.updateBindings)
		}

		// Get bindings
		getPod := &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: scenario.getPod, Namespace: "ns"}}
		bindings = cache.GetBindings(getPod, scenario.getNode)
		if !reflect.DeepEqual(bindings, scenario.getBindings) {
			t.Errorf("Test %v failed: unexpected bindings returned. Got %+v, expected %+v", name, bindings, scenario.updateBindings)
		}
	}
}

func TestDeleteBindings(t *testing.T) {
	initialBindings := []*bindingInfo{{pvc: &v1.PersistentVolumeClaim{ObjectMeta: metav1.ObjectMeta{Name: "pvc1"}}}}
	cache := NewPodBindingCache()

	pod := &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "pod1", Namespace: "ns"}}

	// Get nil bindings
	bindings := cache.GetBindings(pod, "node1")
	if bindings != nil {
		t.Errorf("Test failed: expected initial nil bindings, got %+v", bindings)
	}

	// Delete nothing
	cache.DeleteBindings(pod)

	// Perform updates
	cache.UpdateBindings(pod, "node1", initialBindings)

	// Get bindings
	bindings = cache.GetBindings(pod, "node1")
	if !reflect.DeepEqual(bindings, initialBindings) {
		t.Errorf("Test failed: expected bindings %+v, got %+v", initialBindings, bindings)
	}

	// Delete
	cache.DeleteBindings(pod)

	// Get bindings
	bindings = cache.GetBindings(pod, "node1")
	if bindings != nil {
		t.Errorf("Test failed: expected nil bindings, got %+v", bindings)
	}
}
