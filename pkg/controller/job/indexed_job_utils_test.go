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
	"math"
	"strconv"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	batch "k8s.io/api/batch/v1"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/klog/v2/ktesting"
	"k8s.io/kubernetes/pkg/controller"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/utils/ptr"
)

const noIndex = "-"

func TestCalculateSucceededIndexes(t *testing.T) {
	logger, _ := ktesting.NewTestContext(t)
	cases := map[string]struct {
		prevSucceeded       string
		pods                []indexPhase
		completions         int32
		wantStatusIntervals orderedIntervals
		wantIntervals       orderedIntervals
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
			prevSucceeded:       "0-5,8-10",
			completions:         8,
			wantStatusIntervals: []interval{{0, 5}},
			wantIntervals:       []interval{{0, 5}},
		},
		"prev interval partially out of range": {
			prevSucceeded:       "0-5,8-10",
			completions:         10,
			wantStatusIntervals: []interval{{0, 5}, {8, 9}},
			wantIntervals:       []interval{{0, 5}, {8, 9}},
		},
		"prev and new separate": {
			prevSucceeded: "0,4,5,10-12",
			pods: []indexPhase{
				{"2", v1.PodSucceeded},
				{"7", v1.PodSucceeded},
				{"8", v1.PodSucceeded},
			},
			completions: 13,
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
			completions: 9,
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
			completions: 9,
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
			completions: 10,
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
			completions: 4,
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
			job := &batch.Job{
				Status: batch.JobStatus{
					CompletedIndexes: tc.prevSucceeded,
				},
				Spec: batch.JobSpec{
					Completions: ptr.To(tc.completions),
				},
			}
			pods := hollowPodsWithIndexPhase(tc.pods)
			for _, p := range pods {
				p.Finalizers = append(p.Finalizers, batch.JobTrackingFinalizer)
			}
			gotStatusIntervals, gotIntervals := calculateSucceededIndexes(logger, job, pods)
			if diff := cmp.Diff(tc.wantStatusIntervals, gotStatusIntervals); diff != "" {
				t.Errorf("Unexpected completed indexes from status (-want,+got):\n%s", diff)
			}
			if diff := cmp.Diff(tc.wantIntervals, gotIntervals); diff != "" {
				t.Errorf("Unexpected completed indexes (-want,+got):\n%s", diff)
			}
		})
	}
}

func TestIsIndexFailed(t *testing.T) {
	logger, _ := ktesting.NewTestContext(t)
	cases := map[string]struct {
		job        batch.Job
		pod        *v1.Pod
		wantResult bool
	}{
		"failed pod exceeding backoffLimitPerIndex, when backoffLimitPerIndex=0": {
			job: batch.Job{
				Spec: batch.JobSpec{
					Completions:          ptr.To[int32](2),
					BackoffLimitPerIndex: ptr.To[int32](0),
				},
			},
			pod:        buildPod().indexFailureCount("0").phase(v1.PodFailed).index("0").trackingFinalizer().Pod,
			wantResult: true,
		},
		"failed pod exceeding backoffLimitPerIndex, when backoffLimitPerIndex=1": {
			job: batch.Job{
				Spec: batch.JobSpec{
					Completions:          ptr.To[int32](2),
					BackoffLimitPerIndex: ptr.To[int32](1),
				},
			},
			pod:        buildPod().indexFailureCount("1").phase(v1.PodFailed).index("1").trackingFinalizer().Pod,
			wantResult: true,
		},
		"matching FailIndex pod failure policy": {
			job: batch.Job{
				Spec: batch.JobSpec{
					Completions:          ptr.To[int32](2),
					BackoffLimitPerIndex: ptr.To[int32](1),
					PodFailurePolicy: &batch.PodFailurePolicy{
						Rules: []batch.PodFailurePolicyRule{
							{
								Action: batch.PodFailurePolicyActionFailIndex,
								OnExitCodes: &batch.PodFailurePolicyOnExitCodesRequirement{
									Operator: batch.PodFailurePolicyOnExitCodesOpIn,
									Values:   []int32{3},
								},
							},
						},
					},
				},
			},
			pod: buildPod().indexFailureCount("0").status(v1.PodStatus{
				Phase: v1.PodFailed,
				ContainerStatuses: []v1.ContainerStatus{
					{
						State: v1.ContainerState{
							Terminated: &v1.ContainerStateTerminated{
								ExitCode: 3,
							},
						},
					},
				},
			}).index("0").trackingFinalizer().Pod,
			wantResult: true,
		},
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, feature.DefaultFeatureGate, features.JobBackoffLimitPerIndex, true)
			gotResult := isIndexFailed(logger, &tc.job, tc.pod)
			if diff := cmp.Diff(tc.wantResult, gotResult); diff != "" {
				t.Errorf("Unexpected result (-want,+got):\n%s", diff)
			}
		})
	}
}

func TestCalculateFailedIndexes(t *testing.T) {
	logger, _ := ktesting.NewTestContext(t)
	cases := map[string]struct {
		job                   batch.Job
		pods                  []*v1.Pod
		wantPrevFailedIndexes orderedIntervals
		wantFailedIndexes     orderedIntervals
	}{
		"one new index failed": {
			job: batch.Job{
				Spec: batch.JobSpec{
					Completions:          ptr.To[int32](2),
					BackoffLimitPerIndex: ptr.To[int32](1),
				},
			},
			pods: []*v1.Pod{
				buildPod().indexFailureCount("0").phase(v1.PodFailed).index("0").trackingFinalizer().Pod,
				buildPod().indexFailureCount("1").phase(v1.PodFailed).index("1").trackingFinalizer().Pod,
			},
			wantFailedIndexes: []interval{{1, 1}},
		},
		"pod without finalizer is ignored": {
			job: batch.Job{
				Spec: batch.JobSpec{
					Completions:          ptr.To[int32](2),
					BackoffLimitPerIndex: ptr.To[int32](0),
				},
			},
			pods: []*v1.Pod{
				buildPod().indexFailureCount("0").phase(v1.PodFailed).index("0").Pod,
			},
			wantFailedIndexes: nil,
		},
		"pod outside completions is ignored": {
			job: batch.Job{
				Spec: batch.JobSpec{
					Completions:          ptr.To[int32](2),
					BackoffLimitPerIndex: ptr.To[int32](0),
				},
			},
			pods: []*v1.Pod{
				buildPod().indexFailureCount("0").phase(v1.PodFailed).index("3").Pod,
			},
			wantFailedIndexes: nil,
		},
		"extend the failed indexes": {
			job: batch.Job{
				Status: batch.JobStatus{
					FailedIndexes: ptr.To("0"),
				},
				Spec: batch.JobSpec{
					Completions:          ptr.To[int32](2),
					BackoffLimitPerIndex: ptr.To[int32](0),
				},
			},
			pods: []*v1.Pod{
				buildPod().indexFailureCount("0").phase(v1.PodFailed).index("1").trackingFinalizer().Pod,
			},
			wantFailedIndexes: []interval{{0, 1}},
		},
		"prev failed indexes empty": {
			job: batch.Job{
				Status: batch.JobStatus{
					FailedIndexes: ptr.To(""),
				},
				Spec: batch.JobSpec{
					Completions:          ptr.To[int32](2),
					BackoffLimitPerIndex: ptr.To[int32](0),
				},
			},
			pods: []*v1.Pod{
				buildPod().indexFailureCount("0").phase(v1.PodFailed).index("1").trackingFinalizer().Pod,
			},
			wantFailedIndexes: []interval{{1, 1}},
		},
		"prev failed indexes outside the completions": {
			job: batch.Job{
				Status: batch.JobStatus{
					FailedIndexes: ptr.To("9"),
				},
				Spec: batch.JobSpec{
					Completions:          ptr.To[int32](2),
					BackoffLimitPerIndex: ptr.To[int32](0),
				},
			},
			pods: []*v1.Pod{
				buildPod().indexFailureCount("0").phase(v1.PodFailed).index("1").trackingFinalizer().Pod,
			},
			wantFailedIndexes: []interval{{1, 1}},
		},
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			failedIndexes := calculateFailedIndexes(logger, &tc.job, tc.pods)
			if diff := cmp.Diff(&tc.wantFailedIndexes, failedIndexes); diff != "" {
				t.Errorf("Unexpected failed indexes (-want,+got):\n%s", diff)
			}
		})
	}
}

func TestGetPodsWithDelayedDeletionPerIndex(t *testing.T) {
	logger, _ := ktesting.NewTestContext(t)
	now := time.Now()
	cases := map[string]struct {
		job                                 batch.Job
		pods                                []*v1.Pod
		expectedRmFinalizers                sets.Set[string]
		wantPodsWithDelayedDeletionPerIndex []string
	}{
		"failed pods are kept corresponding to non-failed indexes are kept": {
			job: batch.Job{
				Spec: batch.JobSpec{
					Completions:          ptr.To[int32](3),
					BackoffLimitPerIndex: ptr.To[int32](1),
				},
			},
			pods: []*v1.Pod{
				buildPod().uid("a").indexFailureCount("0").phase(v1.PodFailed).index("0").trackingFinalizer().Pod,
				buildPod().uid("b").indexFailureCount("1").phase(v1.PodFailed).index("1").trackingFinalizer().Pod,
				buildPod().uid("c").indexFailureCount("0").phase(v1.PodFailed).index("2").trackingFinalizer().Pod,
			},
			wantPodsWithDelayedDeletionPerIndex: []string{"a", "c"},
		},
		"failed pod without finalizer; the pod's deletion is not delayed as it already started": {
			job: batch.Job{
				Spec: batch.JobSpec{
					Completions:          ptr.To[int32](2),
					BackoffLimitPerIndex: ptr.To[int32](0),
				},
			},
			pods: []*v1.Pod{
				buildPod().uid("a").indexFailureCount("0").phase(v1.PodFailed).index("0").Pod,
			},
			wantPodsWithDelayedDeletionPerIndex: []string{},
		},
		"failed pod with expected finalizer removal; the pod's deletion is not delayed as it already started": {
			job: batch.Job{
				Spec: batch.JobSpec{
					Completions:          ptr.To[int32](2),
					BackoffLimitPerIndex: ptr.To[int32](0),
				},
			},
			pods: []*v1.Pod{
				buildPod().uid("a").indexFailureCount("0").phase(v1.PodFailed).index("0").trackingFinalizer().Pod,
			},
			expectedRmFinalizers:                sets.New("a"),
			wantPodsWithDelayedDeletionPerIndex: []string{},
		},
		"failed pod with index outside of completions; the pod's deletion is not delayed": {
			job: batch.Job{
				Spec: batch.JobSpec{
					Completions:          ptr.To[int32](2),
					BackoffLimitPerIndex: ptr.To[int32](0),
				},
			},
			pods: []*v1.Pod{
				buildPod().uid("a").indexFailureCount("0").phase(v1.PodFailed).index("4").trackingFinalizer().Pod,
			},
			wantPodsWithDelayedDeletionPerIndex: []string{},
		},
		"failed pod for active index; the pod's deletion is not delayed as it is already replaced": {
			job: batch.Job{
				Spec: batch.JobSpec{
					Completions:          ptr.To[int32](2),
					BackoffLimitPerIndex: ptr.To[int32](1),
				},
			},
			pods: []*v1.Pod{
				buildPod().uid("a1").indexFailureCount("0").phase(v1.PodFailed).index("0").trackingFinalizer().Pod,
				buildPod().uid("a2").indexFailureCount("1").phase(v1.PodRunning).index("0").trackingFinalizer().Pod,
			},
			wantPodsWithDelayedDeletionPerIndex: []string{},
		},
		"failed pod for succeeded index; the pod's deletion is not delayed as it is already replaced": {
			job: batch.Job{
				Spec: batch.JobSpec{
					Completions:          ptr.To[int32](2),
					BackoffLimitPerIndex: ptr.To[int32](1),
				},
			},
			pods: []*v1.Pod{
				buildPod().uid("a1").indexFailureCount("0").phase(v1.PodFailed).index("0").trackingFinalizer().Pod,
				buildPod().uid("a2").indexFailureCount("1").phase(v1.PodSucceeded).index("0").trackingFinalizer().Pod,
			},
			wantPodsWithDelayedDeletionPerIndex: []string{},
		},
		"multiple failed pods for index with different failure count; only the pod with highest failure count is kept": {
			job: batch.Job{
				Spec: batch.JobSpec{
					Completions:          ptr.To[int32](2),
					BackoffLimitPerIndex: ptr.To[int32](4),
				},
			},
			pods: []*v1.Pod{
				buildPod().uid("a1").indexFailureCount("0").phase(v1.PodFailed).index("0").trackingFinalizer().Pod,
				buildPod().uid("a3").indexFailureCount("2").phase(v1.PodFailed).index("0").trackingFinalizer().Pod,
				buildPod().uid("a2").indexFailureCount("1").phase(v1.PodFailed).index("0").trackingFinalizer().Pod,
			},
			wantPodsWithDelayedDeletionPerIndex: []string{"a3"},
		},
		"multiple failed pods for index with different finish times; only the last failed pod is kept": {
			job: batch.Job{
				Spec: batch.JobSpec{
					Completions:          ptr.To[int32](2),
					BackoffLimitPerIndex: ptr.To[int32](4),
				},
			},
			pods: []*v1.Pod{
				buildPod().uid("a1").indexFailureCount("1").phase(v1.PodFailed).index("0").customDeletionTimestamp(now.Add(-time.Second)).trackingFinalizer().Pod,
				buildPod().uid("a3").indexFailureCount("1").phase(v1.PodFailed).index("0").customDeletionTimestamp(now).trackingFinalizer().Pod,
				buildPod().uid("a2").indexFailureCount("1").phase(v1.PodFailed).index("0").customDeletionTimestamp(now.Add(-2 * time.Second)).trackingFinalizer().Pod,
			},
			wantPodsWithDelayedDeletionPerIndex: []string{"a3"},
		},
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, feature.DefaultFeatureGate, features.JobBackoffLimitPerIndex, true)
			activePods := controller.FilterActivePods(logger, tc.pods)
			failedIndexes := calculateFailedIndexes(logger, &tc.job, tc.pods)
			_, succeededIndexes := calculateSucceededIndexes(logger, &tc.job, tc.pods)
			jobCtx := &syncJobCtx{
				job:                  &tc.job,
				pods:                 tc.pods,
				activePods:           activePods,
				succeededIndexes:     succeededIndexes,
				failedIndexes:        failedIndexes,
				expectedRmFinalizers: tc.expectedRmFinalizers,
			}
			gotPodsWithDelayedDeletionPerIndex := getPodsWithDelayedDeletionPerIndex(logger, jobCtx)
			gotPodsWithDelayedDeletionPerIndexSet := sets.New[string]()
			for _, pod := range gotPodsWithDelayedDeletionPerIndex {
				gotPodsWithDelayedDeletionPerIndexSet.Insert(string(pod.UID))
			}
			if diff := cmp.Diff(tc.wantPodsWithDelayedDeletionPerIndex, sets.List(gotPodsWithDelayedDeletionPerIndexSet)); diff != "" {
				t.Errorf("Unexpected set of pods with delayed deletion (-want,+got):\n%s", diff)
			}
		})
	}
}

func TestGetNewIndexFailureCountValue(t *testing.T) {
	logger, _ := ktesting.NewTestContext(t)
	cases := map[string]struct {
		job                             batch.Job
		pod                             *v1.Pod
		wantNewIndexFailureCount        int32
		wantNewIndexIgnoredFailureCount int32
	}{
		"first pod created": {
			job:                      batch.Job{},
			wantNewIndexFailureCount: 0,
		},
		"failed pod being replaced with 0 index failure count": {
			job:                      batch.Job{},
			pod:                      buildPod().uid("a").indexFailureCount("0").phase(v1.PodFailed).index("0").trackingFinalizer().Pod,
			wantNewIndexFailureCount: 1,
		},
		"failed pod being replaced with >0 index failure count": {
			job:                      batch.Job{},
			pod:                      buildPod().uid("a").indexFailureCount("3").phase(v1.PodFailed).index("0").trackingFinalizer().Pod,
			wantNewIndexFailureCount: 4,
		},
		"failed pod being replaced, matching the ignore rule": {
			job: batch.Job{
				Spec: batch.JobSpec{
					PodFailurePolicy: &batch.PodFailurePolicy{
						Rules: []batch.PodFailurePolicyRule{
							{
								Action: batch.PodFailurePolicyActionIgnore,
								OnPodConditions: []batch.PodFailurePolicyOnPodConditionsPattern{
									{
										Type:   v1.DisruptionTarget,
										Status: v1.ConditionTrue,
									},
								},
							},
						},
					},
				},
			},
			pod: buildPod().uid("a").indexFailureCount("3").status(v1.PodStatus{
				Phase: v1.PodFailed,
				Conditions: []v1.PodCondition{
					{
						Type:   v1.DisruptionTarget,
						Status: v1.ConditionTrue,
					},
				},
			}).index("3").trackingFinalizer().Pod,
			wantNewIndexFailureCount:        3,
			wantNewIndexIgnoredFailureCount: 1,
		},
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, feature.DefaultFeatureGate, features.JobBackoffLimitPerIndex, true)
			gotNewIndexFailureCount, gotNewIndexIgnoredFailureCount := getNewIndexFailureCounts(logger, &tc.job, tc.pod)
			if diff := cmp.Diff(tc.wantNewIndexFailureCount, gotNewIndexFailureCount); diff != "" {
				t.Errorf("Unexpected set of pods with delayed deletion (-want,+got):\n%s", diff)
			}
			if diff := cmp.Diff(tc.wantNewIndexIgnoredFailureCount, gotNewIndexIgnoredFailureCount); diff != "" {
				t.Errorf("Unexpected set of pods with delayed deletion (-want,+got):\n%s", diff)
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
		failedIndexes    *orderedIntervals
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
		"with failed indexes": {
			activePods: []indexPhase{
				{"3", v1.PodPending},
				{"9", v1.PodPending},
			},
			succeededIndexes: []interval{{1, 1}, {5, 5}, {9, 9}},
			failedIndexes:    &orderedIntervals{{2, 2}, {6, 7}},
			cnt:              5,
			completions:      20,
			want:             []int{0, 4, 8, 10, 11},
		},
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			jobCtx := &syncJobCtx{
				activePods:       hollowPodsWithIndexPhase(tc.activePods),
				succeededIndexes: tc.succeededIndexes,
				failedIndexes:    tc.failedIndexes,
				job:              newJob(1, 1, 1, batch.IndexedCompletion),
			}
			got := firstPendingIndexes(jobCtx, tc.cnt, tc.completions)
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

func TestGetIndexFailureCount(t *testing.T) {
	logger, _ := ktesting.NewTestContext(t)
	cases := map[string]struct {
		pod        *v1.Pod
		wantResult int32
	}{
		"no annotation": {
			pod:        &v1.Pod{},
			wantResult: 0,
		},
		"valid value": {
			pod:        buildPod().indexFailureCount("2").Pod,
			wantResult: 2,
		},
		"valid maxint32 value": {
			pod:        buildPod().indexFailureCount(strconv.FormatInt(math.MaxInt32, 10)).Pod,
			wantResult: math.MaxInt32,
		},
		"too large value": {
			pod:        buildPod().indexFailureCount(strconv.FormatInt(math.MaxInt32+1, 10)).Pod,
			wantResult: 0,
		},
		"negative value": {
			pod:        buildPod().indexFailureCount("-1").Pod,
			wantResult: 0,
		},
		"invalid int value": {
			pod:        buildPod().indexFailureCount("xyz").Pod,
			wantResult: 0,
		},
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			gotResult := getIndexFailureCount(logger, tc.pod)
			if diff := cmp.Equal(tc.wantResult, gotResult); !diff {
				t.Errorf("Unexpected result. want: %d, got: %d", tc.wantResult, gotResult)
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
