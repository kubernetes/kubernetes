/*
Copyright 2018 The Kubernetes Authors.

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
	"errors"
	"reflect"
	"testing"

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func TestUpdateMatchings(t *testing.T) {
	testcases := map[string]struct {
		updatePod      string
		updatePVC      string
		updateMatching *matching

		getPod      string
		getPVC      string
		getMatching *matching
	}{
		"matching-exists": {
			updatePod:      "pod1",
			updatePVC:      "pvc1",
			updateMatching: &matching{err: errors.New("test")},
			getPod:         "pod1",
			getPVC:         "pvc1",
			getMatching:    &matching{err: errors.New("test")},
		},
		"another-get-pvc": {
			updatePod:      "pod1",
			updatePVC:      "pvc1",
			updateMatching: &matching{err: errors.New("test")},
			getPod:         "pod1",
			getPVC:         "pvc2",
			getMatching:    nil,
		},
	}

	for name, v := range testcases {
		cache := NewPVCMatchingCache()

		// update
		updatePod := &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: v.updatePod, Namespace: "ns"}}
		updatePVC := &v1.PersistentVolumeClaim{ObjectMeta: metav1.ObjectMeta{Name: v.updatePVC, Namespace: "ns"}}
		cache.UpdateMatching(updatePod, updatePVC, v.updateMatching)

		// verify update
		matching := cache.GetMatching(updatePod, updatePVC)
		if !reflect.DeepEqual(matching, v.updateMatching) {
			t.Errorf("Test %v failed: returned matching after update different. Got %+v, expected %+v", name, matching, v.updateMatching)
		}

		// verify get
		getPod := &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: v.getPod, Namespace: "ns"}}
		getPVC := &v1.PersistentVolumeClaim{ObjectMeta: metav1.ObjectMeta{Name: v.getPVC, Namespace: "ns"}}
		getMatching := cache.GetMatching(getPod, getPVC)
		if !reflect.DeepEqual(getMatching, v.getMatching) {
			t.Errorf("Test %v failed: returned matching after get different. Got %+v, expected %+v", name, getMatching, v.getMatching)
		}
	}
}

func TestDeleteMatchings(t *testing.T) {
	initialMatching := &matching{err: nil, pv: &v1.PersistentVolume{ObjectMeta: metav1.ObjectMeta{Name: "pv1"}}}
	cache := NewPVCMatchingCache()
	pod := &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "pod1", Namespace: "ns"}}
	pvc := &v1.PersistentVolumeClaim{ObjectMeta: metav1.ObjectMeta{Name: "pvc1", Namespace: "ns"}}

	// Get nil matching.
	matching := cache.GetMatching(pod, pvc)
	if matching != nil {
		t.Errorf("Test failed: expect initla nil matching, got %+v", matching)
	}

	// Delete nothing.
	cache.DeleteMatchings(pod)

	// Perform update.
	cache.UpdateMatching(pod, pvc, initialMatching)

	// Verify.
	matching = cache.GetMatching(pod, pvc)
	if !reflect.DeepEqual(matching, initialMatching) {
		t.Errorf("Test failed: expected matching %+v, got %+v", initialMatching, matching)
	}

	// Delete
	cache.DeleteMatchings(pod)

	// Verify after delete.
	matching = cache.GetMatching(pod, pvc)
	if matching != nil {
		t.Errorf("Test failed: expected nil matching, got %+v", matching)
	}
}
