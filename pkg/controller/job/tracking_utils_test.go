/*
Copyright 2020 The Kubernetes Authors.

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
	"fmt"
	"sync"
	"testing"

	"github.com/google/go-cmp/cmp"
	batch "k8s.io/api/batch/v1"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/component-base/metrics/testutil"
	"k8s.io/klog/v2/ktesting"
	"k8s.io/kubernetes/pkg/controller/job/metrics"
)

func TestUIDTrackingExpectations(t *testing.T) {
	logger, _ := ktesting.NewTestContext(t)
	tracks := []struct {
		job         string
		firstRound  []string
		secondRound []string
	}{
		{
			job:         "foo",
			firstRound:  []string{"a", "b", "c", "d"},
			secondRound: []string{"e", "f"},
		},
		{
			job:         "bar",
			firstRound:  []string{"x", "y", "z"},
			secondRound: []string{"u", "v", "w"},
		},
		{
			job:         "baz",
			firstRound:  []string{"w"},
			secondRound: []string{"a"},
		},
	}
	expectations := newUIDTrackingExpectations()

	// Insert first round of keys in parallel.

	var wg sync.WaitGroup
	wg.Add(len(tracks))
	errs := make([]error, len(tracks))
	for i := range tracks {
		track := tracks[i]
		go func(errID int) {
			errs[errID] = expectations.expectFinalizersRemoved(logger, track.job, track.firstRound)
			wg.Done()
		}(i)
	}
	wg.Wait()
	for i, err := range errs {
		if err != nil {
			t.Errorf("Failed adding first round of UIDs for job %s: %v", tracks[i].job, err)
		}
	}

	for _, track := range tracks {
		uids := expectations.getSet(track.job)
		if uids == nil {
			t.Errorf("Set of UIDs is empty for job %s", track.job)
		} else if diff := cmp.Diff(track.firstRound, sets.List(uids.set)); diff != "" {
			t.Errorf("Unexpected keys for job %s (-want,+got):\n%s", track.job, diff)
		}
	}

	// Delete the first round of keys and add the second round in parallel.

	for i, track := range tracks {
		wg.Add(len(track.firstRound) + 1)
		track := track
		for _, uid := range track.firstRound {
			uid := uid
			go func() {
				expectations.finalizerRemovalObserved(logger, track.job, uid)
				wg.Done()
			}()
		}
		go func(errID int) {
			errs[errID] = expectations.expectFinalizersRemoved(logger, track.job, track.secondRound)
			wg.Done()
		}(i)
	}
	wg.Wait()

	for i, err := range errs {
		if err != nil {
			t.Errorf("Failed adding second round of UIDs for job %s: %v", tracks[i].job, err)
		}
	}

	for _, track := range tracks {
		uids := expectations.getSet(track.job)
		if uids == nil {
			t.Errorf("Set of UIDs is empty for job %s", track.job)
		} else if diff := cmp.Diff(track.secondRound, sets.List(uids.set)); diff != "" {
			t.Errorf("Unexpected keys for job %s (-want,+got):\n%s", track.job, diff)
		}
	}
	for _, track := range tracks {
		expectations.deleteExpectations(logger, track.job)
		uids := expectations.getSet(track.job)
		if uids != nil {
			t.Errorf("Wanted expectations for job %s to be cleared, but they were not", track.job)
		}
	}
}

func TestRecordFinishedPodWithTrackingFinalizer(t *testing.T) {
	metrics.Register()
	cases := map[string]struct {
		oldPod     *v1.Pod
		newPod     *v1.Pod
		wantAdd    int
		wantDelete int
	}{
		"new non-finished Pod with finalizer": {
			newPod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Finalizers: []string{batch.JobTrackingFinalizer},
				},
				Status: v1.PodStatus{
					Phase: v1.PodPending,
				},
			},
		},
		"pod with finalizer fails": {
			oldPod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Finalizers: []string{batch.JobTrackingFinalizer},
				},
				Status: v1.PodStatus{
					Phase: v1.PodRunning,
				},
			},
			newPod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Finalizers: []string{batch.JobTrackingFinalizer},
				},
				Status: v1.PodStatus{
					Phase: v1.PodFailed,
				},
			},
			wantAdd: 1,
		},
		"pod with finalizer succeeds": {
			oldPod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Finalizers: []string{batch.JobTrackingFinalizer},
				},
				Status: v1.PodStatus{
					Phase: v1.PodRunning,
				},
			},
			newPod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Finalizers: []string{batch.JobTrackingFinalizer},
				},
				Status: v1.PodStatus{
					Phase: v1.PodSucceeded,
				},
			},
			wantAdd: 1,
		},
		"succeeded pod loses finalizer": {
			oldPod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Finalizers: []string{batch.JobTrackingFinalizer},
				},
				Status: v1.PodStatus{
					Phase: v1.PodSucceeded,
				},
			},
			newPod: &v1.Pod{
				Status: v1.PodStatus{
					Phase: v1.PodSucceeded,
				},
			},
			wantDelete: 1,
		},
		"pod without finalizer removed": {
			oldPod: &v1.Pod{
				Status: v1.PodStatus{
					Phase: v1.PodSucceeded,
				},
			},
		},
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			metrics.TerminatedPodsTrackingFinalizerTotal.Reset()
			recordFinishedPodWithTrackingFinalizer(tc.oldPod, tc.newPod)
			if err := validateTerminatedPodsTrackingFinalizerTotal(metrics.Add, tc.wantAdd); err != nil {
				t.Errorf("Failed validating terminated_pods_tracking_finalizer_total(add): %v", err)
			}
			if err := validateTerminatedPodsTrackingFinalizerTotal(metrics.Delete, tc.wantDelete); err != nil {
				t.Errorf("Failed validating terminated_pods_tracking_finalizer_total(delete): %v", err)
			}
		})
	}
}

func validateTerminatedPodsTrackingFinalizerTotal(event string, want int) error {
	got, err := testutil.GetCounterMetricValue(metrics.TerminatedPodsTrackingFinalizerTotal.WithLabelValues(event))
	if err != nil {
		return err
	}
	if int(got) != want {
		return fmt.Errorf("got value %d, want %d", int(got), want)
	}
	return nil
}
