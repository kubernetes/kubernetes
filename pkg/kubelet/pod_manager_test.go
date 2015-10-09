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

package kubelet

import (
	"reflect"
	"testing"

	"k8s.io/kubernetes/pkg/api"
	kubetypes "k8s.io/kubernetes/pkg/kubelet/types"
)

// Stub out mirror client for testing purpose.
func newFakePodManager() (*basicPodManager, *fakeMirrorClient) {
	podManager := newBasicPodManager(nil)
	fakeMirrorClient := newFakeMirrorClient()
	podManager.mirrorClient = fakeMirrorClient
	return podManager, fakeMirrorClient
}

// Tests that pods/maps are properly set after the pod update, and the basic
// methods work correctly.
func TestGetSetPods(t *testing.T) {
	mirrorPod := &api.Pod{
		ObjectMeta: api.ObjectMeta{
			UID:       "987654321",
			Name:      "bar",
			Namespace: "default",
			Annotations: map[string]string{
				kubetypes.ConfigSourceAnnotationKey: "api",
				kubetypes.ConfigMirrorAnnotationKey: "mirror",
			},
		},
	}
	staticPod := &api.Pod{
		ObjectMeta: api.ObjectMeta{
			UID:         "123456789",
			Name:        "bar",
			Namespace:   "default",
			Annotations: map[string]string{kubetypes.ConfigSourceAnnotationKey: "file"},
		},
	}

	expectedPods := []*api.Pod{
		{
			ObjectMeta: api.ObjectMeta{
				UID:         "999999999",
				Name:        "taco",
				Namespace:   "default",
				Annotations: map[string]string{kubetypes.ConfigSourceAnnotationKey: "api"},
			},
		},
		staticPod,
	}
	updates := append(expectedPods, mirrorPod)
	podManager, _ := newFakePodManager()
	podManager.SetPods(updates)

	// Tests that all regular pods are recorded corrrectly.
	actualPods := podManager.GetPods()
	if len(actualPods) != len(expectedPods) {
		t.Errorf("expected %d pods, got %d pods; expected pods %#v, got pods %#v", len(expectedPods), len(actualPods),
			expectedPods, actualPods)
	}
	for _, expected := range expectedPods {
		found := false
		for _, actual := range actualPods {
			if actual.UID == expected.UID {
				if !reflect.DeepEqual(&expected, &actual) {
					t.Errorf("pod was recorded incorrectly. expect: %#v, got: %#v", expected, actual)
				}
				found = true
				break
			}
		}
		if !found {
			t.Errorf("pod %q was not found in %#v", expected.UID, actualPods)
		}
	}
	// Tests UID translation works as expected.
	if uid := podManager.TranslatePodUID(mirrorPod.UID); uid != staticPod.UID {
		t.Errorf("unable to translate UID %q to the static POD's UID %q; %#v",
			mirrorPod.UID, staticPod.UID, podManager.mirrorPodByUID)
	}

	// Test the basic Get methods.
	actualPod, ok := podManager.GetPodByFullName("bar_default")
	if !ok || !reflect.DeepEqual(actualPod, staticPod) {
		t.Errorf("unable to get pod by full name; expected: %#v, got: %#v", staticPod, actualPod)
	}
	actualPod, ok = podManager.GetPodByName("default", "bar")
	if !ok || !reflect.DeepEqual(actualPod, staticPod) {
		t.Errorf("unable to get pod by name; expected: %#v, got: %#v", staticPod, actualPod)
	}

}
