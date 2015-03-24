/*
Copyright 2015 Google Inc. All rights reserved.

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

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
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
	mirrorPod := api.Pod{
		ObjectMeta: api.ObjectMeta{
			UID:       "987654321",
			Name:      "bar",
			Namespace: "default",
			Annotations: map[string]string{
				ConfigSourceAnnotationKey: "api",
				ConfigMirrorAnnotationKey: "mirror",
			},
		},
	}
	staticPod := api.Pod{
		ObjectMeta: api.ObjectMeta{
			UID:         "123456789",
			Name:        "bar",
			Namespace:   "default",
			Annotations: map[string]string{ConfigSourceAnnotationKey: "file"},
		},
	}

	expectedPods := []api.Pod{
		{
			ObjectMeta: api.ObjectMeta{
				UID:         "999999999",
				Name:        "taco",
				Namespace:   "default",
				Annotations: map[string]string{ConfigSourceAnnotationKey: "api"},
			},
		},
		staticPod,
	}
	updates := append(expectedPods, mirrorPod)
	podManager, _ := newFakePodManager()
	podManager.SetPods(updates)
	actualPods := podManager.GetPods()
	if !reflect.DeepEqual(expectedPods, actualPods) {
		t.Errorf("pods are not set correctly; expected %#v, got %#v", expectedPods, actualPods)
	}
	actualPod, ok := podManager.mirrorPodByUID[mirrorPod.UID]
	if !ok {
		t.Errorf("mirror pod %q is not found in the mirror pod map by UID", mirrorPod.UID)
	} else if !reflect.DeepEqual(&mirrorPod, actualPod) {
		t.Errorf("mirror pod is recorded incorrectly. expect: %v, got: %v", mirrorPod, actualPod)
	}
	actualPod, ok = podManager.mirrorPodByFullName[GetPodFullName(&mirrorPod)]
	if !ok {
		t.Errorf("mirror pod %q is not found in the mirror pod map by full name", GetPodFullName(&mirrorPod))
	} else if !reflect.DeepEqual(&mirrorPod, actualPod) {
		t.Errorf("mirror pod is recorded incorrectly. expect: %v, got: %v", mirrorPod, actualPod)
	}
	if uid := podManager.TranslatePodUID(mirrorPod.UID); uid != staticPod.UID {
		t.Errorf("unable to translate UID %q to the static POD's UID %q; %#v", mirrorPod.UID, staticPod.UID, podManager.mirrorPodByUID)
	}
	actualPod, ok = podManager.GetPodByFullName("bar_default")
	if !ok || !reflect.DeepEqual(actualPod, &staticPod) {
		t.Errorf("unable to get pod by full name; expected: %#v, got: %#v", staticPod, actualPod)
	}
	actualPod, ok = podManager.GetPodByName("default", "bar")
	if !ok || !reflect.DeepEqual(actualPod, &staticPod) {
		t.Errorf("unable to get pod by name; expected: %#v, got: %#v", staticPod, actualPod)
	}

}
