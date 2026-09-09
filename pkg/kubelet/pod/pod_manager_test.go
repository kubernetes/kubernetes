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
	"fmt"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	"github.com/stretchr/testify/assert"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	kubetypes "k8s.io/kubernetes/pkg/kubelet/types"
)

var (
	mirrorPodUID  = "mirror-pod-uid"
	staticPodUID  = "static-pod-uid"
	normalPodUID  = "normal-pod-uid"
	mirrorPodName = "mirror-static-pod-name"
	staticPodName = "mirror-static-pod-name"
	normalPodName = "normal-pod-name"

	mirrorPod = &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			UID:       types.UID(mirrorPodUID),
			Name:      mirrorPodName,
			Namespace: metav1.NamespaceDefault,
			Annotations: map[string]string{
				kubetypes.ConfigSourceAnnotationKey: "api",
				kubetypes.ConfigMirrorAnnotationKey: "mirror",
				kubetypes.ConfigHashAnnotationKey:   "mirror",
			},
		},
	}
	staticPod = &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			UID:         types.UID(staticPodUID),
			Name:        staticPodName,
			Namespace:   metav1.NamespaceDefault,
			Annotations: map[string]string{kubetypes.ConfigSourceAnnotationKey: "file"},
		},
	}
	normalPod = &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			UID:         types.UID(normalPodUID),
			Name:        normalPodName,
			Namespace:   metav1.NamespaceDefault,
			Annotations: map[string]string{},
		},
	}
	sortOpt = cmpopts.SortSlices(func(a, b *v1.Pod) bool { return a.Name < b.Name })
)

func TestAddOrUpdatePod(t *testing.T) {
	tests := []struct {
		name     string
		isMirror bool
		pod      *v1.Pod
	}{
		{
			name:     "mirror pod",
			pod:      mirrorPod,
			isMirror: true,
		},
		{
			name:     "static pod",
			pod:      staticPod,
			isMirror: false,
		},
		{
			name:     "normal pod",
			pod:      normalPod,
			isMirror: false,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			pm := &basicManager{
				podByUID:            make(map[kubetypes.ResolvedPodUID]*v1.Pod),
				mirrorPodByUID:      make(map[kubetypes.MirrorPodUID]*v1.Pod),
				podByFullName:       make(map[string]*v1.Pod),
				mirrorPodByFullName: make(map[string]*v1.Pod),
				translationByUID:    make(map[kubetypes.MirrorPodUID]kubetypes.ResolvedPodUID),
			}
			pm.AddPod(test.pod)
			verify(t, pm, test.isMirror, test.pod)
			test.pod.Annotations["testLabel"] = "updated"
			pm.UpdatePod(test.pod)
			verify(t, pm, test.isMirror, test.pod)
		})
	}

}

func verify(t *testing.T, pm *basicManager, isMirror bool, expectedPod *v1.Pod) {
	fullName := fmt.Sprintf("%s_%s", expectedPod.Name, expectedPod.Namespace)
	if isMirror {
		inputPod, ok := pm.mirrorPodByUID[kubetypes.MirrorPodUID(expectedPod.UID)]
		assert.True(t, ok)
		assert.Empty(t, cmp.Diff(expectedPod, inputPod), "MirrorPodByUID map verification failed.")
		inputPod, ok = pm.mirrorPodByFullName[fullName]
		assert.True(t, ok)
		assert.Empty(t, cmp.Diff(expectedPod, inputPod), "MirrorPodByFullName map verification failed.")
	} else {
		inputPod, ok := pm.podByUID[kubetypes.ResolvedPodUID(expectedPod.UID)]
		assert.True(t, ok)
		assert.Empty(t, cmp.Diff(expectedPod, inputPod), "PodByUID map verification failed.")
		inputPod, ok = pm.podByFullName[fullName]
		assert.True(t, ok)
		assert.Empty(t, cmp.Diff(expectedPod, inputPod), "PodByFullName map verification failed.")
	}
}

// Tests that pods/maps are properly set after the pod update, and the basic
// methods work correctly.
func TestGetSetPods(t *testing.T) {
	testCase := []struct {
		name                 string
		podList              []*v1.Pod
		wantUID              types.UID
		wantPod              *v1.Pod
		isMirrorPod          bool
		expectPods           []*v1.Pod
		expectPod            *v1.Pod
		expectedMirrorPod    *v1.Pod
		expectUID            types.UID
		expectPodToMirrorMap map[kubetypes.ResolvedPodUID]kubetypes.MirrorPodUID
		expectMirrorToPodMap map[kubetypes.MirrorPodUID]kubetypes.ResolvedPodUID
	}{
		{
			name:              "Get normal pod",
			podList:           []*v1.Pod{mirrorPod, staticPod, normalPod},
			wantUID:           types.UID("normal-pod-uid"),
			wantPod:           normalPod,
			isMirrorPod:       false,
			expectedMirrorPod: nil,
			expectPods:        []*v1.Pod{staticPod, normalPod},
			expectPod:         normalPod,
			expectUID:         types.UID("normal-pod-uid"),
			expectPodToMirrorMap: map[kubetypes.ResolvedPodUID]kubetypes.MirrorPodUID{
				kubetypes.ResolvedPodUID("static-pod-uid"): kubetypes.MirrorPodUID("mirror-pod-uid"),
			},
			expectMirrorToPodMap: map[kubetypes.MirrorPodUID]kubetypes.ResolvedPodUID{
				kubetypes.MirrorPodUID("mirror-pod-uid"): kubetypes.ResolvedPodUID("static-pod-uid"),
			},
		},
		{
			name:              "Get static pod",
			podList:           []*v1.Pod{mirrorPod, staticPod, normalPod},
			wantUID:           types.UID("static-pod-uid"),
			wantPod:           staticPod,
			isMirrorPod:       false,
			expectPods:        []*v1.Pod{staticPod, normalPod},
			expectPod:         staticPod,
			expectedMirrorPod: mirrorPod,
			expectUID:         types.UID("static-pod-uid"),
			expectPodToMirrorMap: map[kubetypes.ResolvedPodUID]kubetypes.MirrorPodUID{
				kubetypes.ResolvedPodUID("static-pod-uid"): kubetypes.MirrorPodUID("mirror-pod-uid"),
			},
			expectMirrorToPodMap: map[kubetypes.MirrorPodUID]kubetypes.ResolvedPodUID{
				kubetypes.MirrorPodUID("mirror-pod-uid"): kubetypes.ResolvedPodUID("static-pod-uid"),
			},
		},
		{
			name:              "Get mirror pod",
			podList:           []*v1.Pod{mirrorPod, staticPod, normalPod},
			wantUID:           types.UID("static-pod-uid"),
			wantPod:           mirrorPod,
			isMirrorPod:       true,
			expectPods:        []*v1.Pod{staticPod, normalPod},
			expectPod:         staticPod,
			expectedMirrorPod: mirrorPod,
			expectUID:         types.UID("static-pod-uid"),
			expectPodToMirrorMap: map[kubetypes.ResolvedPodUID]kubetypes.MirrorPodUID{
				kubetypes.ResolvedPodUID("static-pod-uid"): kubetypes.MirrorPodUID("mirror-pod-uid"),
			},
			expectMirrorToPodMap: map[kubetypes.MirrorPodUID]kubetypes.ResolvedPodUID{
				kubetypes.MirrorPodUID("mirror-pod-uid"): kubetypes.ResolvedPodUID("static-pod-uid"),
			},
		},
	}
	for _, test := range testCase {
		t.Run(test.name, func(t *testing.T) {
			podManager := NewBasicPodManager()
			podManager.SetPods(test.podList)
			actualPods := podManager.GetPods()
			assert.Empty(t, cmp.Diff(test.expectPods, actualPods, sortOpt), "actualPods and expectPods differ")

			uid := podManager.TranslatePodUID(test.wantPod.UID)
			assert.Equal(t, kubetypes.ResolvedPodUID(test.expectUID), uid, "unable to translate pod UID")

			fullName := fmt.Sprintf("%s_%s", test.wantPod.Name, test.wantPod.Namespace)
			actualPod, ok := podManager.GetPodByFullName(fullName)
			assert.True(t, ok)
			assert.Empty(t, cmp.Diff(test.expectPod, actualPod), "actualPod by full name and expectGetPod differ")

			actualPod, ok = podManager.GetPodByName(test.wantPod.Namespace, test.wantPod.Name)
			assert.True(t, ok)
			assert.Empty(t, cmp.Diff(test.expectPod, actualPod), "actualPod by name and expectGetPod differ")

			actualPod, ok = podManager.GetPodByUID(test.wantUID)
			assert.True(t, ok)
			assert.Empty(t, cmp.Diff(test.expectPod, actualPod), "actualPod and expectGetPod differ")

			podToMirror, mirrorToPod := podManager.GetUIDTranslations()
			assert.Empty(t, cmp.Diff(test.expectPodToMirrorMap, podToMirror), "podToMirror and expectPodToMirror differ")
			assert.Empty(t, cmp.Diff(test.expectMirrorToPodMap, mirrorToPod), "mirrorToPod and expectMirrorToPod differ")

			actualPod, ok = podManager.GetMirrorPodByPod(test.wantPod)
			assert.Equal(t, test.expectedMirrorPod != nil, ok)
			assert.Empty(t, cmp.Diff(test.expectedMirrorPod, actualPod), "actualPod and expectShouldBeMirrorPod differ")

			actualPod, actualMirrorPod, isMirrorPod := podManager.GetPodAndMirrorPod(test.wantPod)
			assert.Equal(t, test.isMirrorPod, isMirrorPod)
			assert.Empty(t, cmp.Diff(test.expectPod, actualPod), "actualPod and expectGetPod differ")
			assert.Empty(t, cmp.Diff(test.expectedMirrorPod, actualMirrorPod), "actualMirrorPod and shouldBeMirrorPod differ")

			isMirrorPod = IsMirrorPodOf(test.wantPod, mirrorPod)
			assert.Equal(t, test.isMirrorPod, isMirrorPod)

			if test.isMirrorPod {
				actualPod, ok = podManager.GetPodByMirrorPod(test.wantPod)
				assert.Equal(t, test.expectedMirrorPod != nil, ok)
				assert.Empty(t, cmp.Diff(test.expectPod, actualPod), "actualPod by name and expectGetPod differ")
			}
		})
	}
}

func TestRemovePods(t *testing.T) {
	testCase := []struct {
		name                             string
		podList                          []*v1.Pod
		needToRemovePod                  *v1.Pod
		expectPods                       []*v1.Pod
		expectMirrorPods                 []*v1.Pod
		expectOrphanedMirrorPodFullnames []string
	}{
		{
			name:             "Remove mirror pod",
			podList:          []*v1.Pod{mirrorPod, staticPod, normalPod},
			needToRemovePod:  mirrorPod,
			expectPods:       []*v1.Pod{normalPod, staticPod},
			expectMirrorPods: []*v1.Pod{},
		},
		{
			name:                             "Remove static pod",
			podList:                          []*v1.Pod{mirrorPod, staticPod, normalPod},
			needToRemovePod:                  staticPod,
			expectPods:                       []*v1.Pod{normalPod},
			expectMirrorPods:                 []*v1.Pod{mirrorPod},
			expectOrphanedMirrorPodFullnames: []string{"mirror-static-pod-name_default"},
		},
		{
			name:             "Remove normal pod",
			podList:          []*v1.Pod{mirrorPod, staticPod, normalPod},
			needToRemovePod:  normalPod,
			expectPods:       []*v1.Pod{staticPod},
			expectMirrorPods: []*v1.Pod{mirrorPod},
		},
	}

	for _, test := range testCase {
		t.Run(test.name, func(t *testing.T) {
			podManager := NewBasicPodManager()
			podManager.SetPods(test.podList)
			podManager.RemovePod(test.needToRemovePod)
			actualPods1 := podManager.GetPods()
			actualPods2, actualMirrorPods, orphanedMirrorPodFullnames := podManager.GetPodsAndMirrorPods()
			// Check if the actual pods and mirror pods match the expected ones.
			assert.Empty(t, cmp.Diff(actualPods1, actualPods2, sortOpt), "actualPods by GetPods() and GetPodsAndMirrorPods() differ")
			assert.Empty(t, cmp.Diff(test.expectPods, actualPods1, sortOpt), "actualPods and expectPods differ")
			assert.Empty(t, cmp.Diff(test.expectMirrorPods, actualMirrorPods, sortOpt), "actualMirrorPods and expectMirrorPods differ")
			assert.Empty(t, cmp.Diff(test.expectOrphanedMirrorPodFullnames, orphanedMirrorPodFullnames), "orphanedMirrorPodFullnames and expectOrphanedMirrorPodFullnames differ")
		})
	}
}

func TestGetStaticPodToMirrorPodMap(t *testing.T) {
	podManager := NewBasicPodManager()
	podManager.SetPods([]*v1.Pod{mirrorPod, staticPod, normalPod})
	m := podManager.GetStaticPodToMirrorPodMap()
	if len(m) != 1 {
		t.Fatalf("GetStaticPodToMirrorPodMap(): got %d static pods, wanted 1 static pod", len(m))
	}
	gotMirrorPod, ok := m[staticPod]
	if !ok {
		t.Fatalf("GetStaticPodToMirrorPodMap(): mirror pod not found for staticPod %v", staticPod)
	}
	if gotMirrorPod.UID != mirrorPod.UID {
		t.Fatalf("GetStaticPodToMirrorPodMap(): got UID %s, want %s", gotMirrorPod.UID, mirrorPod.UID)
	}
}
