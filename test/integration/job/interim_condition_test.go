/*
Copyright The Kubernetes Authors.

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
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	batchv1 "k8s.io/api/batch/v1"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/utils/ptr"
)

// statusUpdateLoss is how the first status update with the interim condition
// is lost. Both are injected in the Job controller's HTTP transport, so the
// controller's client gets a transport error rather than an API status error.
type statusUpdateLoss string

const (
	// The request fails before it reaches the API server, so the update is
	// not applied.
	lossBeforeAPIServer statusUpdateLoss = "update not applied"
	// The API server applies the update, but its response is discarded, so
	// the controller cannot tell whether the update was applied.
	lossAfterAPIServer statusUpdateLoss = "update applied, response lost"
)

type interimConditionTestCase struct {
	job            batchv1.Job
	injectIfBody   []string
	succeedIndex0  bool
	wantActive     int32
	wantCondition  batchv1.JobConditionType
	wantReason     string
	otherCondition batchv1.JobConditionType
}

// TestInterimConditionSurvivesLostStatusUpdate verifies that the outcome of a
// Job is preserved when the first status update that adds the interim
// FailureTarget or SuccessCriteriaMet condition fails, either before or after
// the API server applies it; that no pod is deleted before the API server
// has persisted the condition; and that the Job reaches the terminal
// condition only once its pods are gone. It runs with and without the
// controller's read-your-writes consistency for Jobs, so that the
// resourceVersion preconditions alone are also exercised.
func TestInterimConditionSurvivesLostStatusUpdate(t *testing.T) {
	cases := map[string]interimConditionTestCase{
		"activeDeadlineSeconds": {
			job: batchv1.Job{
				Spec: batchv1.JobSpec{
					ActiveDeadlineSeconds: ptr.To[int64](3),
					BackoffLimit:          ptr.To[int32](0),
				},
			},
			injectIfBody:   []string{string(batchv1.JobFailureTarget), batchv1.JobReasonDeadlineExceeded},
			wantActive:     1,
			wantCondition:  batchv1.JobFailed,
			wantReason:     batchv1.JobReasonDeadlineExceeded,
			otherCondition: batchv1.JobComplete,
		},
		"successPolicy": {
			job: batchv1.Job{
				Spec: batchv1.JobSpec{
					Parallelism:    ptr.To[int32](3),
					Completions:    ptr.To[int32](3),
					CompletionMode: ptr.To(batchv1.IndexedCompletion),
					BackoffLimit:   ptr.To[int32](0),
					SuccessPolicy: &batchv1.SuccessPolicy{
						Rules: []batchv1.SuccessPolicyRule{{SucceededCount: ptr.To[int32](1)}},
					},
				},
			},
			injectIfBody:   []string{string(batchv1.JobSuccessCriteriaMet)},
			succeedIndex0:  true,
			wantActive:     3,
			wantCondition:  batchv1.JobComplete,
			wantReason:     batchv1.JobReasonSuccessPolicy,
			otherCondition: batchv1.JobFailed,
		},
	}
	for name, tc := range cases {
		for _, loss := range []statusUpdateLoss{lossBeforeAPIServer, lossAfterAPIServer} {
			for _, consistency := range []bool{true, false} {
				t.Run(fmt.Sprintf("%s, %s, StaleControllerConsistencyJob=%v", name, loss, consistency), func(t *testing.T) {
					featuregatetesting.SetFeatureGateDuringTest(t, feature.DefaultFeatureGate, features.StaleControllerConsistencyJob, consistency)
					testInterimConditionSurvivesLostStatusUpdate(t, tc, loss)
				})
			}
		}
	}
}

func testInterimConditionSurvivesLostStatusUpdate(t *testing.T, tc interimConditionTestCase, loss statusUpdateLoss) {
	closeFn, restConfig, clientSet, ns := setup(t, "interim-condition")
	t.Cleanup(closeFn)

	var injected atomic.Int32
	// State of the Job controller's requests, guarded by mu:
	// interimPersisted is set once the API server has successfully responded
	// to a status update with the interim condition, whether or not the
	// response reached the controller; any pod DELETE sent before that is
	// recorded in deletesBeforePersisted.
	var (
		mu                     sync.Mutex
		interimPersisted       bool
		lostAsIntended         bool
		deletes                int
		deletesBeforePersisted []string
	)
	restConfig.WrapTransport = func(rt http.RoundTripper) http.RoundTripper {
		return roundTripperFunc(func(req *http.Request) (*http.Response, error) {
			if req.Method == http.MethodDelete && strings.Contains(req.URL.Path, "/pods/") {
				mu.Lock()
				deletes++
				if !interimPersisted {
					deletesBeforePersisted = append(deletesBeforePersisted, req.URL.Path)
				}
				mu.Unlock()
				return rt.RoundTrip(req)
			}
			if req.Method != http.MethodPut || !strings.HasSuffix(req.URL.Path, "/status") || !strings.Contains(req.URL.Path, "/jobs/") || req.Body == nil {
				return rt.RoundTrip(req)
			}
			body, err := io.ReadAll(req.Body)
			if err != nil {
				return nil, err
			}
			req.Body = io.NopCloser(bytes.NewReader(body))
			matches := true
			for _, s := range tc.injectIfBody {
				matches = matches && bytes.Contains(body, []byte(s))
			}
			if matches && injected.CompareAndSwap(0, 1) {
				if loss == lossBeforeAPIServer {
					mu.Lock()
					lostAsIntended = !interimPersisted
					mu.Unlock()
					return nil, errors.New("injected transport error before the request reached the API server")
				}
				resp, err := rt.RoundTrip(req)
				if err != nil {
					return nil, err
				}
				if err := resp.Body.Close(); err != nil {
					return nil, err
				}
				mu.Lock()
				lostAsIntended = !interimPersisted && resp.StatusCode < 300
				if resp.StatusCode < 300 {
					interimPersisted = true
				}
				mu.Unlock()
				return nil, errors.New("injected transport error after the API server applied the request")
			}
			resp, err := rt.RoundTrip(req)
			if matches && err == nil && resp.StatusCode < 300 {
				mu.Lock()
				interimPersisted = true
				mu.Unlock()
			}
			return resp, err
		})
	}
	ctx, cancel := startJobControllerAndWaitForCaches(t, restConfig)
	t.Cleanup(cancel)

	jobObj, err := createJobWithDefaults(ctx, clientSet, ns.Name, &tc.job)
	if err != nil {
		t.Fatalf("Failed to create Job: %v", err)
	}
	waitForPodsToBeActive(ctx, t, clientSet.BatchV1().Jobs(ns.Name), tc.wantActive, jobObj)
	if tc.succeedIndex0 {
		if err := setJobPhaseForIndex(ctx, clientSet, jobObj, v1.PodSucceeded, 0); err != nil {
			t.Fatalf("Failed to mark index 0 succeeded: %v", err)
		}
	}

	var got *batchv1.Job
	err = wait.PollUntilContextTimeout(ctx, 100*time.Millisecond, wait.ForeverTestTimeout, true, func(ctx context.Context) (bool, error) {
		got, err = clientSet.BatchV1().Jobs(ns.Name).Get(ctx, jobObj.Name, metav1.GetOptions{})
		if err != nil {
			return false, err
		}
		return getJobConditionStatus(ctx, got, tc.wantCondition) == v1.ConditionTrue ||
			getJobConditionStatus(ctx, got, tc.otherCondition) == v1.ConditionTrue, nil
	})
	if err != nil {
		t.Fatalf("Waiting for Job to finish: %v", err)
	}
	if injected.Load() == 0 {
		t.Fatal("The status update was never intercepted, so the test did not exercise the failure")
	}
	mu.Lock()
	defer mu.Unlock()
	if !lostAsIntended {
		t.Errorf("The first status update with the interim condition was not lost as intended (%s)", loss)
	}
	if !interimPersisted {
		t.Error("The interim condition was never persisted")
	}
	if deletes == 0 {
		t.Error("The active pods were never deleted")
	}
	if len(deletesBeforePersisted) > 0 {
		t.Errorf("Pods were deleted before the interim condition was persisted: %v", deletesBeforePersisted)
	}
	if ptr.Deref(got.Status.Terminating, 0) != 0 || got.Status.Active != 0 {
		t.Errorf("Job finished with active=%d, terminating=%v", got.Status.Active, got.Status.Terminating)
	}
	conditionTypes := sets.New[batchv1.JobConditionType]()
	for _, c := range got.Status.Conditions {
		if conditionTypes.Has(c.Type) {
			t.Errorf("Job has a duplicate %s condition; conditions: %+v", c.Type, got.Status.Conditions)
		}
		conditionTypes.Insert(c.Type)
	}
	for _, c := range got.Status.Conditions {
		if c.Type == tc.wantCondition && c.Status == v1.ConditionTrue {
			if c.Reason != tc.wantReason {
				t.Errorf("Job finished with %s reason %q, want %q; conditions: %+v", c.Type, c.Reason, tc.wantReason, got.Status.Conditions)
			}
			return
		}
	}
	t.Errorf("Job finished without condition %s; conditions: %+v", tc.wantCondition, got.Status.Conditions)
}
