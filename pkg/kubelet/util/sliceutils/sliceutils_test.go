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
	baseTime := time.Now()
	return []*v1.Pod{
		{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "foo1",
				Namespace: v1.NamespaceDefault,
				UID:       "uid-1",
				CreationTimestamp: metav1.Time{
					Time: baseTime,
				},
			},
		},
		{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "foo2",
				Namespace: v1.NamespaceDefault,
				UID:       "uid-2",
				CreationTimestamp: metav1.Time{
					Time: baseTime.Add(time.Hour * 1),
				},
			},
		},
		{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "foo3",
				Namespace: v1.NamespaceDefault,
				UID:       "uid-3",
				CreationTimestamp: metav1.Time{
					Time: baseTime.Add(time.Hour * 2),
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
		// ascending order by creation time
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

func TestPodsByCreationTimeTieBreaker(t *testing.T) {
	// Test that when CreationTimestamp is equal, UID is used as tie-breaker
	baseTime := time.Now()
	podsWithSameTime := PodsByCreationTime{
		{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "pod-z",
				Namespace: v1.NamespaceDefault,
				UID:       "uid-z",
				CreationTimestamp: metav1.Time{
					Time: baseTime,
				},
			},
		},
		{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "pod-a",
				Namespace: v1.NamespaceDefault,
				UID:       "uid-a",
				CreationTimestamp: metav1.Time{
					Time: baseTime,
				},
			},
		},
		{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "pod-m",
				Namespace: v1.NamespaceDefault,
				UID:       "uid-m",
				CreationTimestamp: metav1.Time{
					Time: baseTime,
				},
			},
		},
	}

	fooTests := []struct {
		pods PodsByCreationTime
		i    int
		j    int
		er   bool
	}{
		// When timestamps are equal, UID comparison determines order
		// uid-a < uid-m < uid-z
		{podsWithSameTime, 0, 1, false}, // uid-z < uid-a? No
		{podsWithSameTime, 1, 0, true},  // uid-a < uid-z? Yes
		{podsWithSameTime, 1, 2, true},  // uid-a < uid-m? Yes
		{podsWithSameTime, 2, 1, false}, // uid-m < uid-a? No
		{podsWithSameTime, 0, 2, false}, // uid-z < uid-m? No
		{podsWithSameTime, 2, 0, true},  // uid-m < uid-z? Yes
	}

	for _, fooTest := range fooTests {
		r := fooTest.pods.Less(fooTest.i, fooTest.j)
		if r != fooTest.er {
			t.Errorf("returned %t but expected %t for pods: %v vs %v (UID: %s vs %s)",
				r, fooTest.er,
				fooTest.pods[fooTest.i].Name, fooTest.pods[fooTest.j].Name,
				fooTest.pods[fooTest.i].UID, fooTest.pods[fooTest.j].UID)
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
