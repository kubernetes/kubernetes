/*
Copyright 2021 The Kubernetes Authors.

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

package job

import (
	"testing"

	"github.com/google/go-cmp/cmp"
	batch "k8s.io/api/batch/v1"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/utils/pointer"
)

const noIndex = "-"

func TestCalculateSucceededIndexes(t *testing.T) {
	cases := map[string]struct {
		prevSucceeded          string
		pods                   []indexPhase
		completions            int32
		trackingWithFinalizers bool
		wantStatusIntervals    orderedIntervals
		wantIntervals          orderedIntervals
	}{
		"one index": {
			pods:          []indexPhase{{"1", v1.PodSucceeded}},
			completions:   2,
			wantIntervals: []interval{{1, 1}},
		},
		"two separate": {
			pods: []indexPhase{
				{"2", v1.PodFailed},
				{"5", v1.PodSucceeded},
				{"5", v1.PodSucceeded},
				{"10", v1.PodFailed},
				{"10", v1.PodSucceeded},
			},
			completions:   11,
			wantIntervals: []interval{{5, 5}, {10, 10}},
		},
		"two intervals": {
			pods: []indexPhase{
				{"0", v1.PodRunning},
				{"1", v1.PodPending},
				{"2", v1.PodSucceeded},
				{"3", v1.PodSucceeded},
				{"5", v1.PodSucceeded},
				{"6", v1.PodSucceeded},
				{"7", v1.PodSucceeded},
			},
			completions:   8,
			wantIntervals: []interval{{2, 3}, {5, 7}},
		},
		"one interval, ignore previous": {
			prevSucceeded: "3-5",
			pods: []indexPhase{
				{"0", v1.PodSucceeded},
				{"1", v1.PodFailed},
				{"1", v1.PodSucceeded},
				{"2", v1.PodSucceeded},
				{"2", v1.PodSucceeded},
				{"3", v1.PodFailed},
			},
			completions:   4,
			wantIntervals: []interval{{0, 2}},
		},
		"one index and one interval": {
			pods: []indexPhase{
				{"0", v1.PodSucceeded},
				{"1", v1.PodFailed},
				{"2", v1.PodSucceeded},
				{"3", v1.PodSucceeded},
				{"4", v1.PodSucceeded},
				{"5", v1.PodSucceeded},
				{noIndex, v1.PodSucceeded},
				{"-2", v1.PodSucceeded},
			},
			completions:   6,
			wantIntervals: []interval{{0, 0}, {2, 5}},
		},
		"out of range": {
			pods: []indexPhase{
				{"0", v1.PodSucceeded},
				{"1", v1.PodSucceeded},
				{"2", v1.PodSucceeded},
				{"3", v1.PodFailed},
				{"4", v1.PodSucceeded},
				{"5", v1.PodSucceeded},
				{noIndex, v1.PodSucceeded},
				{"-2", v1.PodSucceeded},
			},
			completions:   5,
			wantIntervals: []interval{{0, 2}, {4, 4}},
		},
		"prev interval out of range": {
			prevSucceeded:          "0-5,8-10",
			completions:            8,
			trackingWithFinalizers: true,
			wantStatusIntervals:    []interval{{0, 5}},
			wantIntervals:          []interval{{0, 5}},
		},
		"prev interval partially out of range": {
			prevSucceeded:          "0-5,8-10",
			completions:            10,
			trackingWithFinalizers: true,
			wantStatusIntervals:    []interval{{0, 5}, {8, 9}},
			wantIntervals:          []interval{{0, 5}, {8, 9}},
		},
		"prev and new separate": {
			prevSucceeded: "0,4,5,10-12",
			pods: []indexPhase{
				{"2", v1.PodSucceeded},
				{"7", v1.PodSucceeded},
				{"8", v1.PodSucceeded},
			},
			completions:            13,
			trackingWithFinalizers: true,
			wantStatusIntervals: []interval{
				{0, 0},
				{4, 5},
				{10, 12},
			},
			wantIntervals: []interval{
				{0, 0},
				{2, 2},
				{4, 5},
				{7, 8},
				{10, 12},
			},
		},
		"prev between new": {
			prevSucceeded: "3,4,6",
			pods: []indexPhase{
				{"2", v1.PodSucceeded},
				{"7", v1.PodSucceeded},
				{"8", v1.PodSucceeded},
			},
			completions:            9,
			trackingWithFinalizers: true,
			wantStatusIntervals: []interval{
				{3, 4},
				{6, 6},
			},
			wantIntervals: []interval{
				{2, 4},
				{6, 8},
			},
		},
		"new between prev": {
			prevSucceeded: "2,7,8",
			pods: []indexPhase{
				{"3", v1.PodSucceeded},
				{"4", v1.PodSucceeded},
				{"6", v1.PodSucceeded},
			},
			completions:            9,
			trackingWithFinalizers: true,
			wantStatusIntervals: []interval{
				{2, 2},
				{7, 8},
			},
			wantIntervals: []interval{
				{2, 4},
				{6, 8},
			},
		},
		"new within prev": {
			prevSucceeded: "2-7",
			pods: []indexPhase{
				{"0", v1.PodSucceeded},
				{"3", v1.PodSucceeded},
				{"5", v1.PodSucceeded},
				{"9", v1.PodSucceeded},
			},
			completions:            10,
			trackingWithFinalizers: true,
			wantStatusIntervals: []interval{
				{2, 7},
			},
			wantIntervals: []interval{
				{0, 0},
				{2, 7},
				{9, 9},
			},
		},
		"corrupted interval": {
			prevSucceeded: "0,1-foo,bar",
			pods: []indexPhase{
				{"3", v1.PodSucceeded},
			},
			completions:            4,
			trackingWithFinalizers: true,
			wantStatusIntervals: []interval{
				{0, 0},
			},
			wantIntervals: []interval{
				{0, 0},
				{3, 3},
			},
		},
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			defer featuregatetesting.SetFeatureGateDuringTest(t, feature.DefaultFeatureGate, features.JobTrackingWithFinalizers, tc.trackingWithFinalizers)()
			job := &batch.Job{
				ObjectMeta: metav1.ObjectMeta{
					Annotations: map[string]string{
						batch.JobTrackingFinalizer: "",
					},
				},
				Status: batch.JobStatus{
					CompletedIndexes: tc.prevSucceeded,
				},
				Spec: batch.JobSpec{
					Completions: pointer.Int32Ptr(tc.completions),
				},
			}
			pods := hollowPodsWithIndexPhase(tc.pods)
			for _, p := range pods {
				p.Finalizers = append(p.Finalizers, batch.JobTrackingFinalizer)
			}
			gotStatusIntervals, gotIntervals := calculateSucceededIndexes(job, pods)
			if diff := cmp.Diff(tc.wantStatusIntervals, gotStatusIntervals); diff != "" {
				t.Errorf("Unexpected completed indexes from status (-want,+got):\n%s", diff)
			}
			if diff := cmp.Diff(tc.wantIntervals, gotIntervals); diff != "" {
				t.Errorf("Unexpected completed indexes (-want,+got):\n%s", diff)
			}
		})
	}
}

func TestIntervalsHaveIndex(t *testing.T) {
	cases := map[string]struct {
		intervals orderedIntervals
		index     int
		wantHas   bool
	}{
		"empty": {
			index: 4,
		},
		"before all": {
			index:     1,
			intervals: []interval{{2, 4}, {5, 7}},
		},
		"after all": {
			index:     9,
			intervals: []interval{{2, 4}, {6, 8}},
		},
		"in between": {
			index:     5,
			intervals: []interval{{2, 4}, {6, 8}},
		},
		"in first": {
			index:     2,
			intervals: []interval{{2, 4}, {6, 8}},
			wantHas:   true,
		},
		"in second": {
			index:     8,
			intervals: []interval{{2, 4}, {6, 8}},
			wantHas:   true,
		},
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			has := tc.intervals.has(tc.index)
			if has != tc.wantHas {
				t.Errorf("intervalsHaveIndex(_, _) = %t, want %t", has, tc.wantHas)
			}
		})
	}
}

func TestFirstPendingIndexes(t *testing.T) {
	cases := map[string]struct {
		cnt              int
		completions      int
		activePods       []indexPhase
		succeededIndexes []interval
		want             []int
	}{
		"cnt greater than completions": {
			cnt:         5,
			completions: 3,
			want:        []int{0, 1, 2},
		},
		"cnt less than completions": {
			cnt:         2,
			completions: 5,
			want:        []int{0, 1},
		},
		"first pods active": {
			activePods: []indexPhase{
				{"0", v1.PodRunning},
				{"1", v1.PodPending},
			},
			cnt:         3,
			completions: 10,
			want:        []int{2, 3, 4},
		},
		"last pods active or succeeded": {
			activePods: []indexPhase{
				{"6", v1.PodPending},
			},
			succeededIndexes: []interval{{4, 5}},
			cnt:              6,
			completions:      6,
			want:             []int{0, 1, 2, 3},
		},
		"mixed": {
			activePods: []indexPhase{
				{"3", v1.PodPending},
				{"5", v1.PodRunning},
				{"8", v1.PodPending},
				{noIndex, v1.PodRunning},
				{"-3", v1.PodRunning},
			},
			succeededIndexes: []interval{{2, 4}, {9, 9}},
			cnt:              5,
			completions:      20,
			want:             []int{0, 1, 6, 7, 10},
		},
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			pods := hollowPodsWithIndexPhase(tc.activePods)
			got := firstPendingIndexes(pods, tc.succeededIndexes, tc.cnt, tc.completions)
			if diff := cmp.Diff(tc.want, got); diff != "" {
				t.Errorf("Wrong first pending indexes (-want,+got):\n%s", diff)
			}
		})
	}
}

func TestAppendDuplicatedIndexPodsForRemoval(t *testing.T) {
	cases := map[string]struct {
		pods        []indexPhase
		wantRm      []indexPhase
		wantLeft    []indexPhase
		completions int32
	}{
		"all unique": {
			pods: []indexPhase{
				{noIndex, v1.PodPending},
				{"2", v1.PodPending},
				{"5", v1.PodRunning},
				{"6", v1.PodRunning},
			},
			wantRm: []indexPhase{
				{noIndex, v1.PodPending},
				{"6", v1.PodRunning},
			},
			wantLeft: []indexPhase{
				{"2", v1.PodPending},
				{"5", v1.PodRunning},
			},
			completions: 6,
		},
		"all with index": {
			pods: []indexPhase{
				{"5", v1.PodPending},
				{"0", v1.PodRunning},
				{"3", v1.PodPending},
				{"0", v1.PodRunning},
				{"3", v1.PodRunning},
				{"0", v1.PodPending},
				{"6", v1.PodRunning},
				{"6", v1.PodPending},
			},
			wantRm: []indexPhase{
				{"0", v1.PodPending},
				{"0", v1.PodRunning},
				{"3", v1.PodPending},
				{"6", v1.PodRunning},
				{"6", v1.PodPending},
			},
			wantLeft: []indexPhase{
				{"0", v1.PodRunning},
				{"3", v1.PodRunning},
				{"5", v1.PodPending},
			},
			completions: 6,
		},
		"mixed": {
			pods: []indexPhase{
				{noIndex, v1.PodPending},
				{"invalid", v1.PodRunning},
				{"-2", v1.PodRunning},
				{"0", v1.PodPending},
				{"1", v1.PodPending},
				{"1", v1.PodPending},
				{"1", v1.PodRunning},
			},
			wantRm: []indexPhase{
				{noIndex, v1.PodPending},
				{"invalid", v1.PodRunning},
				{"-2", v1.PodRunning},
				{"1", v1.PodPending},
				{"1", v1.PodPending},
			},
			wantLeft: []indexPhase{
				{"0", v1.PodPending},
				{"1", v1.PodRunning},
			},
			completions: 6,
		},
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			pods := hollowPodsWithIndexPhase(tc.pods)
			rm, left := appendDuplicatedIndexPodsForRemoval(nil, nil, pods, int(tc.completions))
			rmInt := toIndexPhases(rm)
			leftInt := toIndexPhases(left)
			if diff := cmp.Diff(tc.wantRm, rmInt); diff != "" {
				t.Errorf("Unexpected pods for removal (-want,+got):\n%s", diff)
			}
			if diff := cmp.Diff(tc.wantLeft, leftInt); diff != "" {
				t.Errorf("Unexpected pods to keep (-want,+got):\n%s", diff)
			}
		})
	}
}

func TestPodGenerateNameWithIndex(t *testing.T) {
	cases := map[string]struct {
		jobname             string
		index               int
		wantPodGenerateName string
	}{
		"short job name": {
			jobname:             "indexed-job",
			index:               1,
			wantPodGenerateName: "indexed-job-1-",
		},
		"job name exceeds MaxGeneneratedNameLength": {
			jobname:             "hhhhhooooohhhhhooooohhhhhooooohhhhhooooohhhhhooooohhhhhooooohhhhhooooo",
			index:               1,
			wantPodGenerateName: "hhhhhooooohhhhhooooohhhhhooooohhhhhooooohhhhhooooohhhhh-1-",
		},
		"job name with index suffix exceeds MaxGeneratedNameLength": {
			jobname:             "hhhhhooooohhhhhooooohhhhhooooohhhhhooooohhhhhooooohhhhhoo",
			index:               1,
			wantPodGenerateName: "hhhhhooooohhhhhooooohhhhhooooohhhhhooooohhhhhooooohhhhh-1-",
		},
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			podGenerateName := podGenerateNameWithIndex(tc.jobname, tc.index)
			if diff := cmp.Equal(tc.wantPodGenerateName, podGenerateName); !diff {
				t.Errorf("Got pod generateName %s, want %s", podGenerateName, tc.wantPodGenerateName)
			}
		})
	}
}

func hollowPodsWithIndexPhase(descs []indexPhase) []*v1.Pod {
	pods := make([]*v1.Pod, 0, len(descs))
	for _, desc := range descs {
		p := &v1.Pod{
			Status: v1.PodStatus{
				Phase: desc.Phase,
			},
		}
		if desc.Index != noIndex {
			p.Annotations = map[string]string{
				batch.JobCompletionIndexAnnotation: desc.Index,
			}
		}
		pods = append(pods, p)
	}
	return pods
}

type indexPhase struct {
	Index string
	Phase v1.PodPhase
}

func toIndexPhases(pods []*v1.Pod) []indexPhase {
	result := make([]indexPhase, len(pods))
	for i, p := range pods {
		index := noIndex
		if p.Annotations != nil {
			index = p.Annotations[batch.JobCompletionIndexAnnotation]
		}
		result[i] = indexPhase{index, p.Status.Phase}
	}
	return result
}
