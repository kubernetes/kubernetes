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
	"sort"
	"testing"

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/kubernetes/pkg/kubelet/configmap"
	podtest "k8s.io/kubernetes/pkg/kubelet/pod/testing"
	"k8s.io/kubernetes/pkg/kubelet/secret"
	kubetypes "k8s.io/kubernetes/pkg/kubelet/types"
)

type podType int

const (
	Static = iota
	Mirror
	Extra
)

// Stub out mirror client for testing purpose.
func newTestManager() (*basicManager, *podtest.FakeMirrorClient) {
	fakeMirrorClient := podtest.NewFakeMirrorClient()
	secretManager := secret.NewFakeManager()
	configMapManager := configmap.NewFakeManager()
	manager := NewBasicPodManager(fakeMirrorClient, secretManager, configMapManager).(*basicManager)
	return manager, fakeMirrorClient
}

func newPod(uid types.UID, name string, namespace string, podtype podType) *v1.Pod {
	ret := v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			UID:         uid,
			Name:        name,
			Namespace:   namespace,
			Annotations: map[string]string{},
		},
	}
	if podtype == Static {
		ret.Annotations[kubetypes.ConfigSourceAnnotationKey] = "file"
		return &ret
	}
	if podtype == Mirror {
		ret.Annotations[kubetypes.ConfigMirrorAnnotationKey] = "mirror"
	}
	ret.Annotations[kubetypes.ConfigSourceAnnotationKey] = "api"
	return &ret
}

// Tests that pods/maps are properly set after the pod update, and the basic
// methods work correctly.
func TestGetSetPods(t *testing.T) {
	mirrorPod := newPod(types.UID("987654321"), "bar", "default", Mirror)
	staticPod := newPod(types.UID("123456789"), "bar", "default", Static)

	expectedPods := []*v1.Pod{newPod(types.UID("999999999"), "taco", "default", Extra), staticPod}
	updates := append(expectedPods, mirrorPod)
	podManager, _ := newTestManager()
	podManager.SetPods(updates)

	// Tests that all regular pods are recorded correctly.
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
	// Tests UID translation works as expected. Convert static pod UID for comparison only.
	if uid := podManager.TranslatePodUID(mirrorPod.UID); uid != kubetypes.ResolvedPodUID(staticPod.UID) {
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

func TestDeletePods(t *testing.T) {
	mirrorPod := newPod(types.UID("mirror-pod-uid"), "mirror-static-pod-name", metav1.NamespaceDefault, Mirror)
	staticPod := newPod(types.UID("static-pod-uid"), "mirror-static-pod-name", metav1.NamespaceDefault, Static)

	expectedPods := []*v1.Pod{
		newPod(types.UID("extra-pod-uid"), "extra-pod-name", metav1.NamespaceDefault, Extra),
		staticPod,
	}
	updates := append(expectedPods, mirrorPod)
	podManager, _ := newTestManager()
	podManager.SetPods(updates)

	podManager.DeletePod(staticPod)

	actualPods := podManager.GetPods()
	if len(actualPods) == len(expectedPods) {
		t.Fatalf("Run DeletePod() error, expected %d pods, got %d pods; ", len(expectedPods)-1, len(actualPods))
	}

	orphanedMirrorPodNames := podManager.GetOrphanedMirrorPodNames()
	expectedOrphanedMirrorPodNameNum := 1
	if len(orphanedMirrorPodNames) != expectedOrphanedMirrorPodNameNum {
		t.Fatalf("Run getOrphanedMirrorPodNames() error, expected %d orphaned mirror pods, got %d orphaned mirror pods; ", expectedOrphanedMirrorPodNameNum, len(orphanedMirrorPodNames))
	}

	expectedOrphanedMirrorPodName := mirrorPod.Name + "_" + mirrorPod.Namespace
	if orphanedMirrorPodNames[0] != expectedOrphanedMirrorPodName {
		t.Fatalf("Run getOrphanedMirrorPodNames() error, expected orphaned mirror pod name : %s, got orphaned mirror pod name %s; ", expectedOrphanedMirrorPodName, orphanedMirrorPodNames[0])
	}
}

func TestAddPod(t *testing.T) {
	expectedPod := newPod(types.UID("extra-pod-uid"), "extra-pod-name", metav1.NamespaceDefault, Extra)

	podManager, _ := newTestManager()
	podManager.AddPod(expectedPod)

	actual := podManager.GetPods()
	if len(actual) != 1 {
		t.Errorf("Unexpected pod count. actual: %d, expected: 1", len(actual))
	}

	if !reflect.DeepEqual(actual[0], expectedPod) {
		t.Errorf("actual: %#v, expected: %#v", actual[0], *expectedPod)
	}
}

func TestGetPodsAndMirrorPods(t *testing.T) {
	expectedPods := []*v1.Pod{
		newPod("123456789", "bar", "default", Extra),
		newPod("999999999", "taco", "default", Static),
	}
	expectedMirrors := []*v1.Pod{
		newPod("987654321", "bar", "default", Mirror),
	}
	updates := append(expectedPods, expectedMirrors...)

	podManager, _ := newTestManager()
	podManager.SetPods(updates)

	actualPods, actualMirrors := podManager.GetPodsAndMirrorPods()

	sort.Slice(actualPods, func(i, j int) bool { return actualPods[i].UID < actualPods[j].UID })
	sort.Slice(actualMirrors, func(i, j int) bool { return actualPods[i].UID < actualPods[j].UID })

	if !reflect.DeepEqual(actualPods, expectedPods) {
		t.Errorf("actualPods: %#v, expectedPods: %#v", actualPods, expectedPods)
	}
	if !reflect.DeepEqual(actualMirrors, expectedMirrors) {
		t.Errorf("actualPods: %#v, expectedPods: %#v", actualMirrors, expectedMirrors)
	}
}

func TestGetPodById(t *testing.T) {
	pods := []*v1.Pod{
		newPod(types.UID("999999999"), "taco", "default", Static),
		newPod(types.UID("123456789"), "bar", "default", Extra),
		newPod(types.UID("987654321"), "bar", "default", Mirror),
	}

	podManager, _ := newTestManager()
	podManager.SetPods(pods)

	testcases := []struct {
		name        string
		keyUID      types.UID
		expectedUID types.UID
		ok          bool
	}{{
		name:        "get pod",
		keyUID:      types.UID("123456789"),
		expectedUID: types.UID("123456789"),
		ok:          true,
	}, {
		name:        "get static pod",
		keyUID:      types.UID("999999999"),
		expectedUID: types.UID("999999999"),
		ok:          true,
	}, {
		name:        "mirror pod not to be got",
		keyUID:      types.UID("987654321"),
		expectedUID: types.UID(""),
		ok:          false,
	},
	}

	for _, test := range testcases {
		t.Run(test.name, func(t *testing.T) {
			actualPod, actualOk := podManager.GetPodByUID(test.keyUID)

			if actualOk != test.ok {
				t.Errorf("actual OK: %t, expected OK: %t", actualOk, test.ok)
			}
			if actualOk && actualPod.UID != test.expectedUID {
				t.Errorf("actual UID: %s, expected UID: %s", actualPod.UID, test.expectedUID)
			}
		})
	}
}

func TestGetUIDTranslations(t *testing.T) {
	staticPod := newPod(types.UID("987654321"), "bar", "default", Static)

	testcases := []struct {
		name                string
		translationByUID    map[kubetypes.MirrorPodUID]kubetypes.ResolvedPodUID
		podByUID            map[kubetypes.ResolvedPodUID]*v1.Pod
		expectedPodToMirror map[kubetypes.ResolvedPodUID]kubetypes.MirrorPodUID
		expectedMirrorToPod map[kubetypes.MirrorPodUID]kubetypes.ResolvedPodUID
	}{{
		name: "test when all static pods have corresponding mirror pods",
		translationByUID: map[kubetypes.MirrorPodUID]kubetypes.ResolvedPodUID{
			kubetypes.MirrorPodUID("mirror1"): kubetypes.ResolvedPodUID("resolved1"),
			kubetypes.MirrorPodUID("mirror2"): kubetypes.ResolvedPodUID("resolved2"),
		},
		podByUID: map[kubetypes.ResolvedPodUID]*v1.Pod{
			kubetypes.ResolvedPodUID("resolved1"): staticPod,
			kubetypes.ResolvedPodUID("resolved2"): staticPod,
		},
		expectedPodToMirror: map[kubetypes.ResolvedPodUID]kubetypes.MirrorPodUID{
			kubetypes.ResolvedPodUID("resolved1"): kubetypes.MirrorPodUID("mirror1"),
			kubetypes.ResolvedPodUID("resolved2"): kubetypes.MirrorPodUID("mirror2"),
		},
		expectedMirrorToPod: map[kubetypes.MirrorPodUID]kubetypes.ResolvedPodUID{
			kubetypes.MirrorPodUID("mirror1"): kubetypes.ResolvedPodUID("resolved1"),
			kubetypes.MirrorPodUID("mirror2"): kubetypes.ResolvedPodUID("resolved2"),
		},
	}, {
		name: "test when a static pod has no mirror pod",
		translationByUID: map[kubetypes.MirrorPodUID]kubetypes.ResolvedPodUID{
			kubetypes.MirrorPodUID("mirror1"): kubetypes.ResolvedPodUID("resolved1"),
			kubetypes.MirrorPodUID("mirror2"): kubetypes.ResolvedPodUID("resolved2"),
		},
		podByUID: map[kubetypes.ResolvedPodUID]*v1.Pod{
			kubetypes.ResolvedPodUID("resolved1"): staticPod,
			kubetypes.ResolvedPodUID("resolved2"): staticPod,
			kubetypes.ResolvedPodUID("resolved3"): staticPod,
		},
		expectedPodToMirror: map[kubetypes.ResolvedPodUID]kubetypes.MirrorPodUID{
			kubetypes.ResolvedPodUID("resolved1"): kubetypes.MirrorPodUID("mirror1"),
			kubetypes.ResolvedPodUID("resolved2"): kubetypes.MirrorPodUID("mirror2"),
			kubetypes.ResolvedPodUID("resolved3"): kubetypes.MirrorPodUID(""),
		},
		expectedMirrorToPod: map[kubetypes.MirrorPodUID]kubetypes.ResolvedPodUID{
			kubetypes.MirrorPodUID("mirror1"): kubetypes.ResolvedPodUID("resolved1"),
			kubetypes.MirrorPodUID("mirror2"): kubetypes.ResolvedPodUID("resolved2"),
		},
	}}

	for _, test := range testcases {
		t.Run(test.name, func(t *testing.T) {
			podManager, _ := newTestManager()
			podManager.translationByUID = test.translationByUID
			podManager.podByUID = test.podByUID
			actualPodToMirror, actualMirrorToPod := podManager.GetUIDTranslations()

			if !reflect.DeepEqual(actualPodToMirror, test.expectedPodToMirror) {
				t.Errorf("actual PodToMirror: %#v, expected PodToMirror: %#v", actualPodToMirror, test.expectedPodToMirror)
			}
			if !reflect.DeepEqual(actualMirrorToPod, test.expectedMirrorToPod) {
				t.Errorf("actual MirrorToPod: %#v, expected MirrorToPod: %#v", actualMirrorToPod, test.expectedMirrorToPod)
			}
		})
	}
}

func TestIsMirrorPodOf(t *testing.T) {
	testcases := []struct {
		name      string
		pod       *v1.Pod
		mirrorPod *v1.Pod
		expected  bool
	}{{
		name: "is mirror",
		pod: &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				UID:         types.UID("123456789"),
				Name:        "bar",
				Namespace:   "default",
				Annotations: map[string]string{kubetypes.ConfigHashAnnotationKey: "test"},
			},
		},
		mirrorPod: &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				UID:         types.UID("987654321"),
				Name:        "bar",
				Namespace:   "default",
				Annotations: map[string]string{kubetypes.ConfigMirrorAnnotationKey: "test"},
			},
		},
		expected: true,
	}, {
		name: "different annotation",
		pod: &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				UID:         types.UID("123456789"),
				Name:        "bar",
				Namespace:   "default",
				Annotations: map[string]string{kubetypes.ConfigHashAnnotationKey: "test"},
			},
		},
		mirrorPod: &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				UID:         types.UID("987654321"),
				Name:        "bar",
				Namespace:   "default",
				Annotations: map[string]string{kubetypes.ConfigMirrorAnnotationKey: "test2"},
			},
		},
		expected: false,
	}, {
		name: "different name",
		pod: &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				UID:         types.UID("123456789"),
				Name:        "bar",
				Namespace:   "default",
				Annotations: map[string]string{kubetypes.ConfigHashAnnotationKey: "test"},
			},
		},
		mirrorPod: &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				UID:         types.UID("987654321"),
				Name:        "baz",
				Namespace:   "default",
				Annotations: map[string]string{kubetypes.ConfigMirrorAnnotationKey: "test"},
			},
		},
		expected: false,
	}}

	for _, test := range testcases {
		t.Run(test.name, func(t *testing.T) {
			podManager, _ := newTestManager()
			actual := podManager.IsMirrorPodOf(test.mirrorPod, test.pod)

			if actual != test.expected {
				t.Errorf("actual: %t, expected: %t", actual, test.expected)
			}
		})
	}
}

func TestGetMirrorPodByPod(t *testing.T) {
	testcases := []struct {
		name              string
		pod               *v1.Pod
		expectedMirrorPod *v1.Pod
		expectedOk        bool
	}{{
		name:              "test pod found",
		pod:               newPod(types.UID("987654321"), "bar", "default", Static),
		expectedMirrorPod: newPod(types.UID("123456789"), "bar", "default", Mirror),
		expectedOk:        true,
	}, {
		name:              "test pod not found",
		pod:               newPod(types.UID("987654321"), "bar", "default", Static),
		expectedMirrorPod: newPod(types.UID("123456789"), "baz", "default", Mirror),
		expectedOk:        false,
	}}

	for _, test := range testcases {
		t.Run(test.name, func(t *testing.T) {
			podManager, _ := newTestManager()
			podManager.AddPod(test.expectedMirrorPod)
			actualMirrorPod, actualOk := podManager.GetMirrorPodByPod(test.pod)

			if actualOk != test.expectedOk {
				t.Errorf("actualOk: %t, expectedOk: %t", actualOk, test.expectedOk)
			}

			if actualOk && !reflect.DeepEqual(actualMirrorPod, test.expectedMirrorPod) {
				t.Errorf("actualMirrorPod: %#v, expectedMirrorPod: %#v", actualMirrorPod, test.expectedMirrorPod)
			}
		})
	}
}

func TestGetPodByMirrorPod(t *testing.T) {
	testcases := []struct {
		name        string
		mirrorPod   *v1.Pod
		expectedPod *v1.Pod
		expectedOk  bool
	}{{
		name:        "test pod found",
		mirrorPod:   newPod(types.UID("987654321"), "bar", "default", Mirror),
		expectedPod: newPod(types.UID("123456789"), "bar", "default", Static),
		expectedOk:  true,
	}, {
		name:        "test pod not found",
		mirrorPod:   newPod(types.UID("987654321"), "bar", "default", Mirror),
		expectedPod: newPod(types.UID("123456789"), "baz", "default", Static),
		expectedOk:  false,
	}}

	for _, test := range testcases {
		t.Run(test.name, func(t *testing.T) {
			podManager, _ := newTestManager()
			podManager.AddPod(test.expectedPod)
			actualPod, actualOk := podManager.GetPodByMirrorPod(test.mirrorPod)

			if actualOk != test.expectedOk {
				t.Errorf("actualOk: %t, expectedOk: %t", actualOk, test.expectedOk)
			}

			if actualOk && !reflect.DeepEqual(actualPod, test.expectedPod) {
				t.Errorf("actualMirrorPod: %#v, expectedMirrorPod: %#v", actualPod, test.expectedPod)
			}
		})
	}
}
