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

package resourcepoolstatusrequest

import (
	"context"
	"errors"
	"strings"
	"testing"
	"time"

	resourcev1 "k8s.io/api/resource/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes/fake"
	resourcev1listers "k8s.io/client-go/listers/resource/v1"
	k8stesting "k8s.io/client-go/testing"
	"k8s.io/client-go/tools/record"
	"k8s.io/client-go/util/workqueue"
	basemetrics "k8s.io/component-base/metrics"
	"k8s.io/component-base/metrics/testutil"
	"k8s.io/klog/v2/ktesting"
	"k8s.io/kubernetes/pkg/controller/resourcepoolstatusrequest/metrics"
)

// failingSliceLister makes calculatePoolStatus take its CalculationFailed path.
type failingSliceLister struct{}

func (failingSliceLister) List(labels.Selector) ([]*resourcev1.ResourceSlice, error) {
	return nil, errors.New("injected list failure")
}

func (failingSliceLister) Get(string) (*resourcev1.ResourceSlice, error) {
	return nil, errors.New("injected get failure")
}

var _ resourcev1listers.ResourceSliceLister = failingSliceLister{}

// TestSyncRequestObservability asserts which metrics and events each sync
// outcome produces, so that no path is silent.
func TestSyncRequestObservability(t *testing.T) {
	testCases := map[string]struct {
		slice             *resourcev1.ResourceSlice
		requestDriver     string // defaults to the slice's driver
		retriesExhausted  bool
		updateStatusErr   error
		failListing       bool
		wantErr           error  // nil, errIncompletePool, or errAny
		wantLabel         string // driver_name; defaults to the request's driver
		wantResult        string // empty if nothing was processed
		wantErrors        float64
		wantDurationCount uint64
		wantEventReason   string
	}{
		"complete pool": {
			slice:             makeSlice("slice-1", "complete.example.com", "pool-1", "node-1", 4),
			wantResult:        metrics.ResultComplete,
			wantDurationCount: 1,
		},
		"driver without slices is labeled unknown": {
			slice:             makeSlice("slice-1", "known.example.com", "pool-1", "node-1", 4),
			requestDriver:     "typo.example.com",
			wantLabel:         metrics.UnknownDriver,
			wantResult:        metrics.ResultComplete,
			wantDurationCount: 1,
		},
		"incomplete pool with retries left is requeued silently": {
			slice:   makeSliceWithExpectedCount("slice-1", "requeue.example.com", "pool-1", "node-1", 4, 1, 3),
			wantErr: errIncompletePool,
		},
		"incomplete pool after retries is written and reported": {
			slice:             makeSliceWithExpectedCount("slice-1", "giveup.example.com", "pool-1", "node-1", 4, 1, 3),
			retriesExhausted:  true,
			wantResult:        metrics.ResultIncomplete,
			wantDurationCount: 1,
			wantEventReason:   eventReasonPoolIncomplete,
		},
		// A concurrent modification is retried quietly, not counted as an error.
		"status update conflict": {
			slice:           makeSlice("slice-1", "conflict.example.com", "pool-1", "node-1", 4),
			updateStatusErr: apierrors.NewConflict(schema.GroupResource{Resource: "resourcepoolstatusrequests"}, "test-request", errors.New("modified")),
			wantErr:         errStatusConflict,
		},
		"status update failure": {
			slice:             makeSlice("slice-1", "updatefail.example.com", "pool-1", "node-1", 4),
			updateStatusErr:   errors.New("injected update failure"),
			wantErr:           errAny,
			wantResult:        metrics.ResultError,
			wantErrors:        1,
			wantDurationCount: 1,
		},
		"calculation failure": {
			slice:             makeSlice("slice-1", "listfail.example.com", "pool-1", "node-1", 4),
			failListing:       true,
			wantLabel:         metrics.UnknownDriver,
			wantResult:        metrics.ResultFailed,
			wantDurationCount: 1,
			wantEventReason:   eventReasonCalculationFailed,
		},
	}

	for name, tc := range testCases {
		t.Run(name, func(t *testing.T) {
			_, ctx := ktesting.NewTestContext(t)
			driver := tc.slice.Spec.Driver
			if tc.requestDriver != "" {
				driver = tc.requestDriver
			}
			label := driver
			if tc.wantLabel != "" {
				label = tc.wantLabel
			}
			request := makeRequest(driver)

			fakeClient := fake.NewClientset(request)
			if tc.updateStatusErr != nil {
				fakeClient.PrependReactor("update", "resourcepoolstatusrequests", func(action k8stesting.Action) (bool, runtime.Object, error) {
					return action.GetSubresource() == "status", nil, tc.updateStatusErr
				})
			}
			informerFactory := informers.NewSharedInformerFactory(fakeClient, 0)
			controller, err := NewController(ctx, fakeClient,
				informerFactory.Resource().V1alpha3().ResourcePoolStatusRequests(),
				informerFactory.Resource().V1().ResourceSlices(),
				informerFactory.Resource().V1().ResourceClaims(),
				informerFactory.Resource().V1().DeviceTaintRules(),
			)
			if err != nil {
				t.Fatalf("Failed to create controller: %v", err)
			}
			recorder := record.NewFakeRecorder(10)
			controller.recorder = recorder
			if err := informerFactory.Resource().V1alpha3().ResourcePoolStatusRequests().Informer().GetStore().Add(request); err != nil {
				t.Fatalf("Failed to add request to informer: %v", err)
			}
			if err := informerFactory.Resource().V1().ResourceSlices().Informer().GetStore().Add(tc.slice); err != nil {
				t.Fatalf("Failed to add slice to informer: %v", err)
			}
			if tc.failListing {
				controller.sliceLister = failingSliceLister{}
			}
			if tc.retriesExhausted {
				controller.workqueue.ShutDown()
				controller.workqueue = workqueue.NewTypedRateLimitingQueue(workqueue.NewTypedItemExponentialFailureRateLimiter[string](0, 0))
				for range maxRetries {
					controller.workqueue.AddRateLimited(request.Name)
					key, _ := controller.workqueue.Get()
					controller.workqueue.Done(key)
				}
			}

			results := []string{metrics.ResultComplete, metrics.ResultIncomplete, metrics.ResultFailed, metrics.ResultError}
			processed := map[string]func() float64{}
			for _, result := range results {
				processed[result] = counterDelta(t, metrics.RequestsProcessed.WithLabelValues(label, result))
			}
			errs := counterDelta(t, metrics.RequestProcessingErrors.WithLabelValues(label))
			durations := histogramCountDelta(t, metrics.RequestProcessingDuration.WithLabelValues(label))

			err = controller.syncRequest(ctx, request.Name)
			switch {
			case tc.wantErr == nil && err != nil:
				t.Fatalf("syncRequest: unexpected error %v", err)
			case errors.Is(tc.wantErr, errAny):
				if err == nil {
					t.Fatal("syncRequest: expected an error, got nil")
				}
			case tc.wantErr != nil && !errors.Is(err, tc.wantErr):
				t.Fatalf("syncRequest: got error %v, want %v", err, tc.wantErr)
			}

			for _, result := range results {
				want := 0.0
				if result == tc.wantResult {
					want = 1
				}
				if got := processed[result](); got != want {
					t.Errorf("requests_processed_total{driver_name=%q,result=%q} delta = %v, want %v", label, result, got, want)
				}
			}
			if got := errs(); got != tc.wantErrors {
				t.Errorf("request_processing_errors_total delta = %v, want %v", got, tc.wantErrors)
			}
			if got := durations(); got != tc.wantDurationCount {
				t.Errorf("request_processing_duration_seconds count delta = %v, want %v", got, tc.wantDurationCount)
			}

			var events []string
			for len(recorder.Events) > 0 {
				events = append(events, <-recorder.Events)
			}
			switch {
			case tc.wantEventReason == "" && len(events) > 0:
				t.Errorf("Expected no events, got %v", events)
			case tc.wantEventReason != "" && (len(events) != 1 || !strings.HasPrefix(events[0], "Warning "+tc.wantEventReason+" ")):
				t.Errorf("Expected one Warning %s event, got %v", tc.wantEventReason, events)
			}
		})
	}
}

// A conflict on the final attempt for an incomplete pool must not drop the
// request: the retry does not count against maxRetries.
func TestConflictDoesNotUseRetryBudget(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)
	slice := makeSliceWithExpectedCount("slice-1", "conflict-giveup.example.com", "pool-1", "node-1", 4, 1, 3)
	request := makeRequest(slice.Spec.Driver)

	fakeClient := fake.NewClientset(request)
	conflicts := 1
	fakeClient.PrependReactor("update", "resourcepoolstatusrequests", func(action k8stesting.Action) (bool, runtime.Object, error) {
		if action.GetSubresource() != "status" || conflicts == 0 {
			return false, nil, nil
		}
		conflicts--
		return true, nil, apierrors.NewConflict(schema.GroupResource{Resource: "resourcepoolstatusrequests"}, request.Name, errors.New("modified"))
	})
	informerFactory := informers.NewSharedInformerFactory(fakeClient, 0)
	controller, err := NewController(ctx, fakeClient,
		informerFactory.Resource().V1alpha3().ResourcePoolStatusRequests(),
		informerFactory.Resource().V1().ResourceSlices(),
		informerFactory.Resource().V1().ResourceClaims(),
		informerFactory.Resource().V1().DeviceTaintRules(),
	)
	if err != nil {
		t.Fatalf("Failed to create controller: %v", err)
	}
	if err := informerFactory.Resource().V1alpha3().ResourcePoolStatusRequests().Informer().GetStore().Add(request); err != nil {
		t.Fatalf("Failed to add request to informer: %v", err)
	}
	if err := informerFactory.Resource().V1().ResourceSlices().Informer().GetStore().Add(slice); err != nil {
		t.Fatalf("Failed to add slice to informer: %v", err)
	}
	controller.workqueue.ShutDown()
	controller.workqueue = workqueue.NewTypedRateLimitingQueue(workqueue.NewTypedItemExponentialFailureRateLimiter[string](0, 0))
	defer controller.workqueue.ShutDown()
	for range maxRetries {
		controller.workqueue.AddRateLimited(request.Name)
		key, _ := controller.workqueue.Get()
		controller.workqueue.Done(key)
	}

	// The give-up attempt conflicts, the retry writes the partial status.
	controller.workqueue.Add(request.Name)
	controller.processNextWorkItem(ctx)
	// Checked before the next Get, which would block forever on a dropped key.
	if err := wait.PollUntilContextTimeout(ctx, 10*time.Millisecond, time.Second, true, func(context.Context) (bool, error) {
		return controller.workqueue.Len() > 0, nil
	}); err != nil {
		t.Fatal("Request was dropped after the conflict instead of being retried")
	}
	controller.processNextWorkItem(ctx)
	if got := controller.workqueue.NumRequeues(request.Name); got != 0 {
		t.Errorf("NumRequeues = %d after the status was written, want 0 (forgotten)", got)
	}
	updated, err := fakeClient.ResourceV1alpha3().ResourcePoolStatusRequests().Get(ctx, request.Name, metav1.GetOptions{})
	if err != nil {
		t.Fatalf("Failed to get request: %v", err)
	}
	if updated.Status == nil {
		t.Fatal("Expected the partial status to be written after the conflict, got none")
	}
}

var errAny = errors.New("any error")

func counterDelta(t *testing.T, c basemetrics.CounterMetric) func() float64 {
	t.Helper()
	before, err := testutil.GetCounterMetricValue(c)
	if err != nil {
		t.Fatalf("read counter: %v", err)
	}
	return func() float64 {
		after, err := testutil.GetCounterMetricValue(c)
		if err != nil {
			t.Fatalf("read counter: %v", err)
		}
		return after - before
	}
}

func histogramCountDelta(t *testing.T, m basemetrics.ObserverMetric) func() uint64 {
	t.Helper()
	before, err := testutil.GetHistogramMetricCount(m)
	if err != nil {
		t.Fatalf("read histogram: %v", err)
	}
	return func() uint64 {
		after, err := testutil.GetHistogramMetricCount(m)
		if err != nil {
			t.Fatalf("read histogram: %v", err)
		}
		return after - before
	}
}
