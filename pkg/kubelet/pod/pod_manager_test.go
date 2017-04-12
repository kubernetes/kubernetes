/*
Copyright 2015 The Kubernetes Authors.

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

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/kubernetes/pkg/api/v1"
	podtest "k8s.io/kubernetes/pkg/kubelet/pod/testing"
	"k8s.io/kubernetes/pkg/kubelet/secret"
	kubetypes "k8s.io/kubernetes/pkg/kubelet/types"
)

// Stub out mirror client for testing purpose.
func newTestManager() (*BasicManager, *podtest.FakeMirrorClient) {
	fakeMirrorClient := podtest.NewFakeMirrorClient()
	secretManager := secret.NewFakeManager()
	manager := NewBasicPodManager(fakeMirrorClient, secretManager)
	return manager, fakeMirrorClient
}

// Tests that pods/maps are properly set after the pod update, and the basic
// methods work correctly.
func TestGetSetPods(t *testing.T) {
	apiPods := []*v1.Pod{
		{
			ObjectMeta: metav1.ObjectMeta{
				UID:         "999999999",
				Name:        "taco",
				Namespace:   "default",
				Annotations: map[string]string{kubetypes.ConfigSourceAnnotationKey: "api"},
			},
		},
		{
			ObjectMeta: metav1.ObjectMeta{
				UID:         "123456789",
				Name:        "bar",
				Namespace:   "default",
				Annotations: map[string]string{kubetypes.ConfigSourceAnnotationKey: "file"},
			},
		},
		{
			ObjectMeta: metav1.ObjectMeta{
				UID:       "987654321",
				Name:      "bar",
				Namespace: "default",
				Annotations: map[string]string{
					kubetypes.ConfigSourceAnnotationKey: "api",
					kubetypes.ConfigMirrorAnnotationKey: "mirror",
				},
			},
		},
	}
	podManager, _ := newTestManager()
	pods := podManager.SetPods(apiPods)
	testPod := pods[0]
	staticPod := pods[1]
	mirrorPod := pods[2]
	expectedPods := []*Pod{testPod, staticPod}

	// Tests that all regular pods are recorded correctly.
	actualPods := podManager.GetPods()
	if len(actualPods) != len(expectedPods) {
		t.Errorf("expected %d pods, got %d pods; expected pods %#v, got pods %#v", len(expectedPods), len(actualPods),
			expectedPods, actualPods)
	}
	for _, expected := range expectedPods {
		found := false
		for _, actual := range actualPods {
			if actual.UID() == expected.UID() {
				if !reflect.DeepEqual(&expected, &actual) {
					t.Errorf("pod was recorded incorrectly. expect: %#v, got: %#v", expected, actual)
				}
				found = true
				break
			}
		}
		if !found {
			t.Errorf("pod %q was not found in %#v", expected.UID(), actualPods)
		}
	}
	// Tests UID translation works as expected.
	if uid := podManager.TranslatePodUID(mirrorPod.UID()); uid != staticPod.UID() {
		t.Errorf("unable to translate UID %q to the static POD's UID %q; %#v",
			mirrorPod.UID(), staticPod.UID(), podManager.mirrorPodByUID)
	}

	// Test the basic Get methods.
	actualPod, ok := podManager.GetPodByName("default", "bar")
	if !ok || !reflect.DeepEqual(actualPod, staticPod) {
		t.Errorf("unable to get pod by name; expected: %#v, got: %#v", staticPod.String(), actualPod.String())
	}

}

func TestDeletePods(t *testing.T) {
	apiMirrorPod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			UID:       types.UID("mirror-pod-uid"),
			Name:      "mirror-static-pod-name",
			Namespace: metav1.NamespaceDefault,
			Annotations: map[string]string{
				kubetypes.ConfigSourceAnnotationKey: "api",
				kubetypes.ConfigMirrorAnnotationKey: "mirror",
			},
		},
	}
	mirrorPod := NewPod(apiMirrorPod)
	apiStaticPod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			UID:         types.UID("static-pod-uid"),
			Name:        "mirror-static-pod-name",
			Namespace:   metav1.NamespaceDefault,
			Annotations: map[string]string{kubetypes.ConfigSourceAnnotationKey: "file"},
		},
	}
	staticPod := NewPod(apiStaticPod)

	expectedApiPods := []*v1.Pod{
		{
			ObjectMeta: metav1.ObjectMeta{
				UID:         types.UID("extra-pod-uid"),
				Name:        "extra-pod-name",
				Namespace:   metav1.NamespaceDefault,
				Annotations: map[string]string{kubetypes.ConfigSourceAnnotationKey: "api"},
			},
		},
		apiStaticPod,
	}
	podManager, _ := newTestManager()
	expectedPods := podManager.SetPods(append(expectedApiPods, apiMirrorPod))

	podManager.DeletePod(staticPod)

	actualPods := podManager.GetPods()
	if len(actualPods) == len(expectedPods) {
		t.Fatalf("Run DeletePod() error, expected %d pods, got %d pods; ", len(expectedPods)-1, len(actualPods))
	}

	orphanedMirrorPodNames := podManager.getOrphanedMirrorPodNames()
	expectedOrphanedMirrorPodNameNum := 1
	if len(orphanedMirrorPodNames) != expectedOrphanedMirrorPodNameNum {
		t.Fatalf("Run getOrphanedMirrorPodNames() error, expected %d orphaned mirror pods, got %d orphaned mirror pods; ", expectedOrphanedMirrorPodNameNum, len(orphanedMirrorPodNames))
	}

	expectedOrphanedMirrorPodName := mirrorPod.Name() + "_" + mirrorPod.Namespace()
	if orphanedMirrorPodNames[0] != expectedOrphanedMirrorPodName {
		t.Fatalf("Run getOrphanedMirrorPodNames() error, expected orphaned mirror pod name : %s, got orphaned mirror pod name %s; ", expectedOrphanedMirrorPodName, orphanedMirrorPodNames[0])
	}
}
