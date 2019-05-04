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

package scheduling

import (
	"reflect"
	"testing"

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func TestUpdateGetBindings(t *testing.T) {
	scenarios := map[string]struct {
		updateBindings      []*bindingInfo
		updateProvisionings []*v1.PersistentVolumeClaim
		updatePod           string
		updateNode          string

		getBindings      []*bindingInfo
		getProvisionings []*v1.PersistentVolumeClaim
		getPod           string
		getNode          string
	}{
		"no-pod": {
			getPod:  "pod1",
			getNode: "node1",
		},
		"no-node": {
			updatePod:           "pod1",
			updateNode:          "node1",
			updateBindings:      []*bindingInfo{},
			updateProvisionings: []*v1.PersistentVolumeClaim{},
			getPod:              "pod1",
			getNode:             "node2",
		},
		"binding-nil": {
			updatePod:           "pod1",
			updateNode:          "node1",
			updateBindings:      nil,
			updateProvisionings: nil,
			getPod:              "pod1",
			getNode:             "node1",
		},
		"binding-exists": {
			updatePod:           "pod1",
			updateNode:          "node1",
			updateBindings:      []*bindingInfo{{pvc: &v1.PersistentVolumeClaim{ObjectMeta: metav1.ObjectMeta{Name: "pvc1"}}}},
			updateProvisionings: []*v1.PersistentVolumeClaim{{ObjectMeta: metav1.ObjectMeta{Name: "pvc2"}}},
			getPod:              "pod1",
			getNode:             "node1",
			getBindings:         []*bindingInfo{{pvc: &v1.PersistentVolumeClaim{ObjectMeta: metav1.ObjectMeta{Name: "pvc1"}}}},
			getProvisionings:    []*v1.PersistentVolumeClaim{{ObjectMeta: metav1.ObjectMeta{Name: "pvc2"}}},
		},
	}

	for name, scenario := range scenarios {
		cache := NewPodBindingCache()

		// Perform updates
		updatePod := &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: scenario.updatePod, Namespace: "ns"}}
		cache.UpdateBindings(updatePod, scenario.updateNode, scenario.updateBindings, scenario.updateProvisionings)

		// Verify updated bindings
		bindings := cache.GetBindings(updatePod, scenario.updateNode)
		if !reflect.DeepEqual(bindings, scenario.updateBindings) {
			t.Errorf("Test %v failed: returned bindings after update different. Got %+v, expected %+v", name, bindings, scenario.updateBindings)
		}

		// Verify updated provisionings
		provisionings := cache.GetProvisionedPVCs(updatePod, scenario.updateNode)
		if !reflect.DeepEqual(provisionings, scenario.updateProvisionings) {
			t.Errorf("Test %v failed: returned provisionings after update different. Got %+v, expected %+v", name, provisionings, scenario.updateProvisionings)
		}

		// Get bindings
		getPod := &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: scenario.getPod, Namespace: "ns"}}
		bindings = cache.GetBindings(getPod, scenario.getNode)
		if !reflect.DeepEqual(bindings, scenario.getBindings) {
			t.Errorf("Test %v failed: unexpected bindings returned. Got %+v, expected %+v", name, bindings, scenario.updateBindings)
		}

		// Get provisionings
		provisionings = cache.GetProvisionedPVCs(getPod, scenario.getNode)
		if !reflect.DeepEqual(provisionings, scenario.getProvisionings) {
			t.Errorf("Test %v failed: unexpected bindings returned. Got %+v, expected %+v", name, provisionings, scenario.getProvisionings)
		}
	}
}

func TestDeleteBindings(t *testing.T) {
	initialBindings := []*bindingInfo{{pvc: &v1.PersistentVolumeClaim{ObjectMeta: metav1.ObjectMeta{Name: "pvc1"}}}}
	initialProvisionings := []*v1.PersistentVolumeClaim{{ObjectMeta: metav1.ObjectMeta{Name: "pvc2"}}}
	cache := NewPodBindingCache()

	pod := &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "pod1", Namespace: "ns"}}

	// Get nil bindings and provisionings
	bindings := cache.GetBindings(pod, "node1")
	if bindings != nil {
		t.Errorf("Test failed: expected initial nil bindings, got %+v", bindings)
	}
	provisionings := cache.GetProvisionedPVCs(pod, "node1")
	if provisionings != nil {
		t.Errorf("Test failed: expected initial nil provisionings, got %+v", provisionings)
	}

	// Delete nothing
	cache.DeleteBindings(pod)

	// Perform updates
	cache.UpdateBindings(pod, "node1", initialBindings, initialProvisionings)

	// Get bindings and provisionings
	bindings = cache.GetBindings(pod, "node1")
	if !reflect.DeepEqual(bindings, initialBindings) {
		t.Errorf("Test failed: expected bindings %+v, got %+v", initialBindings, bindings)
	}
	provisionings = cache.GetProvisionedPVCs(pod, "node1")
	if !reflect.DeepEqual(provisionings, initialProvisionings) {
		t.Errorf("Test failed: expected provisionings %+v, got %+v", initialProvisionings, provisionings)
	}

	// Delete
	cache.DeleteBindings(pod)

	// Get bindings and provisionings
	bindings = cache.GetBindings(pod, "node1")
	if bindings != nil {
		t.Errorf("Test failed: expected nil bindings, got %+v", bindings)
	}
	provisionings = cache.GetProvisionedPVCs(pod, "node1")
	if provisionings != nil {
		t.Errorf("Test failed: expected nil provisionings, got %+v", provisionings)
	}
}
