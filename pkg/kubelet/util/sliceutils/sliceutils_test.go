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

package sliceutils

import (
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
)

func buildPodsByCreationTime() PodsByCreationTime {
	return []*v1.Pod{
		{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "foo1",
				Namespace: v1.NamespaceDefault,
				CreationTimestamp: metav1.Time{
					Time: time.Now(),
				},
			},
		},
		{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "foo2",
				Namespace: v1.NamespaceDefault,
				CreationTimestamp: metav1.Time{
					Time: time.Now().Add(time.Hour * 1),
				},
			},
		},
		{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "foo3",
				Namespace: v1.NamespaceDefault,
				CreationTimestamp: metav1.Time{
					Time: time.Now().Add(time.Hour * 2),
				},
			},
		},
	}
}

func TestPodsByCreationTimeLen(t *testing.T) {
	fooTests := []struct {
		pods PodsByCreationTime
		el   int
	}{
		{[]*v1.Pod{}, 0},
		{buildPodsByCreationTime(), 3},
		{[]*v1.Pod{nil, {}}, 2},
		{nil, 0},
	}

	for _, fooTest := range fooTests {
		r := fooTest.pods.Len()
		if r != fooTest.el {
			t.Errorf("returned %d but expected %d for the len of PodsByCreationTime=%s", r, fooTest.el, fooTest.pods)
		}
	}
}

func TestPodsByCreationTimeSwap(t *testing.T) {
	fooTests := []struct {
		pods PodsByCreationTime
		i    int
		j    int
	}{
		{buildPodsByCreationTime(), 0, 1},
		{buildPodsByCreationTime(), 2, 1},
	}

	for _, fooTest := range fooTests {
		fooi := fooTest.pods[fooTest.i]
		fooj := fooTest.pods[fooTest.j]
		fooTest.pods.Swap(fooTest.i, fooTest.j)
		if fooi.GetName() != fooTest.pods[fooTest.j].GetName() || fooj.GetName() != fooTest.pods[fooTest.i].GetName() {
			t.Errorf("failed to swap for %v", fooTest)
		}
	}
}

func TestPodsByCreationTimeLess(t *testing.T) {
	fooTests := []struct {
		pods PodsByCreationTime
		i    int
		j    int
		er   bool
	}{
		// ascending order
		{buildPodsByCreationTime(), 0, 2, true},
		{buildPodsByCreationTime(), 1, 0, false},
	}

	for _, fooTest := range fooTests {
		r := fooTest.pods.Less(fooTest.i, fooTest.j)
		if r != fooTest.er {
			t.Errorf("returned %t but expected %t for the foo=%s", r, fooTest.er, fooTest.pods)
		}
	}
}

func buildByImageSize() ByImageSize {
	return []kubecontainer.Image{
		{
			ID:          "1",
			RepoTags:    []string{"foo-tag11", "foo-tag12"},
			RepoDigests: []string{"foo-rd11", "foo-rd12"},
			Size:        1,
		},
		{
			ID:          "2",
			RepoTags:    []string{"foo-tag21", "foo-tag22"},
			RepoDigests: []string{"foo-rd21", "foo-rd22"},
			Size:        2,
		},
		{
			ID:          "3",
			RepoTags:    []string{"foo-tag31", "foo-tag32"},
			RepoDigests: []string{"foo-rd31", "foo-rd32"},
			Size:        3,
		},
		{
			ID:          "4",
			RepoTags:    []string{"foo-tag41", "foo-tag42"},
			RepoDigests: []string{"foo-rd41", "foo-rd42"},
			Size:        3,
		},
	}
}

func TestByImageSizeLen(t *testing.T) {
	fooTests := []struct {
		images ByImageSize
		el     int
	}{
		{[]kubecontainer.Image{}, 0},
		{buildByImageSize(), 4},
		{nil, 0},
	}

	for _, fooTest := range fooTests {
		r := fooTest.images.Len()
		if r != fooTest.el {
			t.Errorf("returned %d but expected %d for the len of ByImageSize=%v", r, fooTest.el, fooTest.images)
		}
	}
}

func TestByImageSizeSwap(t *testing.T) {
	fooTests := []struct {
		images ByImageSize
		i      int
		j      int
	}{
		{buildByImageSize(), 0, 1},
		{buildByImageSize(), 2, 1},
	}

	for _, fooTest := range fooTests {
		fooi := fooTest.images[fooTest.i]
		fooj := fooTest.images[fooTest.j]
		fooTest.images.Swap(fooTest.i, fooTest.j)
		if fooi.ID != fooTest.images[fooTest.j].ID || fooj.ID != fooTest.images[fooTest.i].ID {
			t.Errorf("failed to swap for %v", fooTest)
		}
	}
}

func TestByImageSizeLess(t *testing.T) {
	fooTests := []struct {
		images ByImageSize
		i      int
		j      int
		er     bool
	}{
		// descending order
		{buildByImageSize(), 0, 2, false},
		{buildByImageSize(), 1, 0, true},
		{buildByImageSize(), 3, 2, true},
	}

	for _, fooTest := range fooTests {
		r := fooTest.images.Less(fooTest.i, fooTest.j)
		if r != fooTest.er {
			t.Errorf("returned %t but expected %t for the foo=%v", r, fooTest.er, fooTest.images)
		}
	}
}

func buildPodsByPriority() PodsByPriority {
	helperPriorities := []int32{
		int32(200),
		int32(100),
	}
	return []*v1.Pod{
		{
			Spec: v1.PodSpec{
				Priority: &helperPriorities[0],
			},
			ObjectMeta: metav1.ObjectMeta{
				Name:      "critical1",
				Namespace: v1.NamespaceDefault,
				CreationTimestamp: metav1.Time{
					Time: time.Now(),
				},
			},
		},
		{
			Spec: v1.PodSpec{
				Priority: &helperPriorities[0],
			},
			ObjectMeta: metav1.ObjectMeta{
				Name:      "critical2",
				Namespace: v1.NamespaceDefault,
				CreationTimestamp: metav1.Time{
					Time: time.Now().Add(time.Hour * 1),
				},
			},
		},
		{
			Spec: v1.PodSpec{
				Priority: &helperPriorities[1],
			},
			ObjectMeta: metav1.ObjectMeta{
				Name:      "lowpriority",
				Namespace: v1.NamespaceDefault,
				CreationTimestamp: metav1.Time{
					Time: time.Now(),
				},
			},
		},
		{
			Spec: v1.PodSpec{
				Priority: nil,
			},
			ObjectMeta: metav1.ObjectMeta{
				Name:      "nopriority",
				Namespace: v1.NamespaceDefault,
				CreationTimestamp: metav1.Time{
					Time: time.Now(),
				},
			},
		},
	}
}

func TestPodsByPriorityLen(t *testing.T) {
	fooTests := []struct {
		pods PodsByPriority
		el   int
	}{
		{[]*v1.Pod{}, 0},
		{buildPodsByPriority(), 4},
		{[]*v1.Pod{nil}, 1},
		{nil, 0},
	}

	for _, fooTest := range fooTests {
		r := fooTest.pods.Len()
		if r != fooTest.el {
			t.Errorf("returned %d but expected %d for the len of ByImageSize=%v", r, fooTest.el, fooTest.pods)
		}
	}
}

func TestPodsByPrioritySwap(t *testing.T) {
	fooTests := []struct {
		pods PodsByPriority
		i    int
		j    int
	}{
		{buildPodsByPriority(), 0, 1},
		{buildPodsByPriority(), 0, 2},
	}
	for _, fooTest := range fooTests {
		fooi := fooTest.pods[fooTest.i]
		fooj := fooTest.pods[fooTest.j]
		fooTest.pods.Swap(fooTest.i, fooTest.j)
		if fooi.GetName() != fooTest.pods[fooTest.j].GetName() || fooj.GetName() != fooTest.pods[fooTest.i].GetName() {
			t.Errorf("failed to swap for %v", fooTest)
		}
	}
}

func TestPodsByPriorityLess(t *testing.T) {
	fooTests := []struct {
		pods PodsByPriority
		i    int
		j    int
		er   bool
	}{
		{buildPodsByPriority(), 0, 2, true},
		{buildPodsByPriority(), 2, 1, false},
		{buildPodsByPriority(), 0, 1, true},
		{buildPodsByPriority(), 3, 2, false},
		{buildPodsByPriority(), 0, 3, true},
	}
	for _, fooTest := range fooTests {
		result := PodsByPriority.Less(fooTest.pods, fooTest.i, fooTest.j)
		if result != fooTest.er {
			t.Errorf("returned %t but expected %t for the foo=%v", result, fooTest.er, fooTest.pods)
		}
	}
}

func TestSortPodsByPriorityAndPreviouslyRunningStatus(t *testing.T) {
	now := time.Now()
	prio := func(v int32) *int32 { return &v }

	tests := []struct {
		name              string
		pods              []*v1.Pod
		previouslyRunning map[string]bool
		expectedOrder     []string
	}{
		{
			name: "previously-running pods come before new pods regardless of priority",
			pods: []*v1.Pod{
				{ObjectMeta: metav1.ObjectMeta{Name: "new-high", CreationTimestamp: metav1.Time{Time: now.Add(1 * time.Second)}}, Spec: v1.PodSpec{Priority: prio(1000)}},
				{ObjectMeta: metav1.ObjectMeta{Name: "old-low", CreationTimestamp: metav1.Time{Time: now}}, Spec: v1.PodSpec{Priority: prio(100)}},
			},
			previouslyRunning: map[string]bool{"old-low": true},
			expectedOrder:     []string{"old-low", "new-high"},
		},
		{
			name: "within previously-running group, sorted by priority descending",
			pods: []*v1.Pod{
				{ObjectMeta: metav1.ObjectMeta{Name: "running-low", CreationTimestamp: metav1.Time{Time: now}}, Spec: v1.PodSpec{Priority: prio(100)}},
				{ObjectMeta: metav1.ObjectMeta{Name: "running-high", CreationTimestamp: metav1.Time{Time: now.Add(1 * time.Second)}}, Spec: v1.PodSpec{Priority: prio(1000)}},
				{ObjectMeta: metav1.ObjectMeta{Name: "new-mid", CreationTimestamp: metav1.Time{Time: now.Add(2 * time.Second)}}, Spec: v1.PodSpec{Priority: prio(500)}},
			},
			previouslyRunning: map[string]bool{"running-low": true, "running-high": true},
			expectedOrder:     []string{"running-high", "running-low", "new-mid"},
		},
		{
			name: "within new group, sorted by priority descending",
			pods: []*v1.Pod{
				{ObjectMeta: metav1.ObjectMeta{Name: "new-low", CreationTimestamp: metav1.Time{Time: now}}, Spec: v1.PodSpec{Priority: prio(100)}},
				{ObjectMeta: metav1.ObjectMeta{Name: "new-high", CreationTimestamp: metav1.Time{Time: now.Add(1 * time.Second)}}, Spec: v1.PodSpec{Priority: prio(1000)}},
			},
			previouslyRunning: map[string]bool{},
			expectedOrder:     []string{"new-high", "new-low"},
		},
		{
			name: "all previously running preserves priority order",
			pods: []*v1.Pod{
				{ObjectMeta: metav1.ObjectMeta{Name: "a-low", CreationTimestamp: metav1.Time{Time: now}}, Spec: v1.PodSpec{Priority: prio(100)}},
				{ObjectMeta: metav1.ObjectMeta{Name: "b-high", CreationTimestamp: metav1.Time{Time: now.Add(1 * time.Second)}}, Spec: v1.PodSpec{Priority: prio(1000)}},
			},
			previouslyRunning: map[string]bool{"a-low": true, "b-high": true},
			expectedOrder:     []string{"b-high", "a-low"},
		},
		{
			name: "same priority - previously running before new, then by creation time",
			pods: []*v1.Pod{
				{ObjectMeta: metav1.ObjectMeta{Name: "new-a", CreationTimestamp: metav1.Time{Time: now}}, Spec: v1.PodSpec{Priority: prio(100)}},
				{ObjectMeta: metav1.ObjectMeta{Name: "running-b", CreationTimestamp: metav1.Time{Time: now.Add(1 * time.Second)}}, Spec: v1.PodSpec{Priority: prio(100)}},
				{ObjectMeta: metav1.ObjectMeta{Name: "running-c", CreationTimestamp: metav1.Time{Time: now.Add(2 * time.Second)}}, Spec: v1.PodSpec{Priority: prio(100)}},
			},
			previouslyRunning: map[string]bool{"running-b": true, "running-c": true},
			expectedOrder:     []string{"running-b", "running-c", "new-a"},
		},
		{
			name: "mixed priorities and running states",
			pods: []*v1.Pod{
				{ObjectMeta: metav1.ObjectMeta{Name: "running-low", CreationTimestamp: metav1.Time{Time: now}}, Spec: v1.PodSpec{Priority: prio(100)}},
				{ObjectMeta: metav1.ObjectMeta{Name: "new-critical", CreationTimestamp: metav1.Time{Time: now.Add(3 * time.Second)}}, Spec: v1.PodSpec{Priority: prio(2000000000)}},
				{ObjectMeta: metav1.ObjectMeta{Name: "running-high", CreationTimestamp: metav1.Time{Time: now.Add(1 * time.Second)}}, Spec: v1.PodSpec{Priority: prio(1000)}},
				{ObjectMeta: metav1.ObjectMeta{Name: "new-low", CreationTimestamp: metav1.Time{Time: now.Add(2 * time.Second)}}, Spec: v1.PodSpec{Priority: prio(100)}},
			},
			previouslyRunning: map[string]bool{"running-low": true, "running-high": true},
			// previously-running first (by priority): running-high, running-low
			// then new (by priority): new-critical, new-low
			expectedOrder: []string{"running-high", "running-low", "new-critical", "new-low"},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			wasPreviouslyRunning := func(pod *v1.Pod) bool {
				return tc.previouslyRunning[pod.Name]
			}
			SortPodsByPriorityAndPreviouslyRunningStatus(tc.pods, wasPreviouslyRunning)

			got := make([]string, len(tc.pods))
			for i, p := range tc.pods {
				got[i] = p.Name
			}
			if len(got) != len(tc.expectedOrder) {
				t.Fatalf("expected %v but got %v", tc.expectedOrder, got)
			}
			for i := range got {
				if got[i] != tc.expectedOrder[i] {
					t.Errorf("at index %d: expected %s but got %s (full order: %v)", i, tc.expectedOrder[i], got[i], got)
				}
			}
		})
	}
}
