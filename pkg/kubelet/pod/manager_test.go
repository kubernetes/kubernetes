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

package pod

import (
	"reflect"
	"testing"

	"github.com/stretchr/testify/assert"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/fake"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	kubetypes "k8s.io/kubernetes/pkg/kubelet/types"
)

func newTestManager() *basicManager {
	return NewBasicPodManager(&fake.Clientset{}).(*basicManager)
}

// Tests that pods/maps are properly set after the pod update, and the basic
// methods work correctly.
func TestGetSetPods(t *testing.T) {
	updates := []*api.Pod{
		{
			ObjectMeta: api.ObjectMeta{
				UID:       "123456789",
				Name:      "bar",
				Namespace: "default",
			},
		},
		{
			ObjectMeta: api.ObjectMeta{
				UID:       "999999999",
				Name:      "foo",
				Namespace: "default",
			},
		},
	}
	podManager := newTestManager()
	podManager.SetPods(updates)

	// Tests that all regular pods are recorded correctly.
	expectedPods := updates
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

	verifyGetMethods(t, podManager, updates[0], true)
}

func TestAddDeleteStaticPod(t *testing.T) {
	// Static pod
	staticPod := &api.Pod{
		ObjectMeta: api.ObjectMeta{
			UID:         "987654321",
			Name:        "static",
			Namespace:   "default",
			Annotations: map[string]string{kubetypes.ConfigSourceAnnotationKey: "file"},
		},
	}
	updates := []*api.Pod{
		{
			ObjectMeta: api.ObjectMeta{
				UID:       "123456789",
				Name:      "bar",
				Namespace: "default",
			},
		},
		staticPod,
	}

	podManager := newTestManager()

	// Test static pod are properly set
	podManager.SetPods(updates)
	assert.NoError(t, verifyChannel(podManager.mirrorPodManager, 1, 0))
	verifyGetMethods(t, podManager, staticPod, true)

	// Test static pod are properly deleted
	podManager.DeletePod(staticPod)
	assert.NoError(t, verifyChannel(podManager.mirrorPodManager, 0, 1))
	verifyGetMethods(t, podManager, staticPod, false)
}

// Test the basic Get methods.
func verifyGetMethods(t *testing.T, podManager *basicManager, pod *api.Pod, found bool) {
	verifyPod := func(t *testing.T, actual, expected *api.Pod, ok, found bool) {
		if found && (!ok || !reflect.DeepEqual(actual, expected)) {
			t.Errorf("unable to get pod; expected: %#v, got: %#v", expected, actual)
		}
		if !found && ok {
			t.Errorf("should not find pod; expected: %#v, got: %#v", expected, actual)
		}
	}
	actualPod, ok := podManager.GetPodByFullName(kubecontainer.GetPodFullName(pod))
	verifyPod(t, actualPod, pod, ok, found)
	actualPod, ok = podManager.GetPodByName(pod.Namespace, pod.Name)
	verifyPod(t, actualPod, pod, ok, found)
	actualPod, ok = podManager.GetPodByUID(pod.UID)
	verifyPod(t, actualPod, pod, ok, found)
}
