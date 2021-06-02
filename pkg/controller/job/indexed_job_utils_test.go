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
)

const noIndex = "-"

func TestCalculateSucceededIndexes(t *testing.T) {
	cases := map[string]struct {
		pods        []indexPhase
		wantCount   int32
		completions int32
	}{
		"1": {
			pods:        []indexPhase{{"1", v1.PodSucceeded}},
			wantCount:   1,
			completions: 2,
		},
		"5,10": {
			pods: []indexPhase{
				{"2", v1.PodFailed},
				{"5", v1.PodSucceeded},
				{"5", v1.PodSucceeded},
				{"10", v1.PodFailed},
				{"10", v1.PodSucceeded},
			},
			wantCount:   2,
			completions: 11,
		},
		"2,3,5-7": {
			pods: []indexPhase{
				{"0", v1.PodRunning},
				{"1", v1.PodPending},
				{"2", v1.PodSucceeded},
				{"3", v1.PodSucceeded},
				{"5", v1.PodSucceeded},
				{"6", v1.PodSucceeded},
				{"7", v1.PodSucceeded},
			},
			wantCount:   5,
			completions: 8,
		},
		"0-2": {
			pods: []indexPhase{
				{"0", v1.PodSucceeded},
				{"1", v1.PodFailed},
				{"1", v1.PodSucceeded},
				{"2", v1.PodSucceeded},
				{"2", v1.PodSucceeded},
				{"3", v1.PodFailed},
			},
			wantCount:   3,
			completions: 4,
		},
		"0,2-5": {
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
			wantCount:   5,
			completions: 6,
		},
		"0-2,4": {
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
			wantCount:   4,
			completions: 5,
		},
	}
	for want, tc := range cases {
		t.Run(want, func(t *testing.T) {
			pods := hollowPodsWithIndexPhase(tc.pods)
			gotStr, gotCnt := calculateSucceededIndexes(pods, tc.completions)
			if diff := cmp.Diff(want, gotStr); diff != "" {
				t.Errorf("Unexpected completed indexes (-want,+got):\n%s", diff)
			}
			if gotCnt != tc.wantCount {
				t.Errorf("Got number of completed indexes %d, want %d", gotCnt, tc.wantCount)
			}
		})
	}
}

func TestFirstPendingIndexes(t *testing.T) {
	cases := map[string]struct {
		cnt         int
		completions int
		pods        []indexPhase
		want        []int
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
		"first pods running or succeeded": {
			pods: []indexPhase{
				{"0", v1.PodRunning},
				{"1", v1.PodPending},
				{"2", v1.PodFailed},
			},
			cnt:         3,
			completions: 10,
			want:        []int{2, 3, 4},
		},
		"last pods running or succeeded": {
			pods: []indexPhase{
				{"4", v1.PodFailed},
				{"5", v1.PodSucceeded},
				{"6", v1.PodPending},
			},
			cnt:         6,
			completions: 6,
			want:        []int{0, 1, 2, 3, 4},
		},
		"mixed": {
			pods: []indexPhase{
				{"1", v1.PodFailed},
				{"2", v1.PodSucceeded},
				{"3", v1.PodPending},
				{"5", v1.PodFailed},
				{"5", v1.PodRunning},
				{"8", v1.PodPending},
				{noIndex, v1.PodRunning},
				{"-3", v1.PodRunning},
			},
			cnt:         5,
			completions: 10,
			want:        []int{0, 1, 4, 6, 7},
		},
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			pods := hollowPodsWithIndexPhase(tc.pods)
			got := firstPendingIndexes(pods, tc.cnt, tc.completions)
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
