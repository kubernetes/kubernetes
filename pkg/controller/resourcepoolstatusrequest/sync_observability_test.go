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
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes/fake"
	k8stesting "k8s.io/client-go/testing"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/tools/record"
	basemetrics "k8s.io/component-base/metrics"
	"k8s.io/component-base/metrics/testutil"
	"k8s.io/klog/v2/ktesting"
	"k8s.io/kubernetes/pkg/controller/resourcepoolstatusrequest/metrics"
)

// failingSliceIndexer forces the CalculationFailed path.
type failingSliceIndexer struct{ cache.Indexer }

func (failingSliceIndexer) ByIndex(string, string) ([]interface{}, error) {
	return nil, errors.New("injected list failure")
}

// TestSyncRequestObservability checks the metrics and events of each sync outcome.
func TestSyncRequestObservability(t *testing.T) {
	conflict := apierrors.NewConflict(schema.GroupResource{Resource: "resourcepoolstatusrequests"}, "test-request", errors.New("modified"))
	testCases := map[string]struct {
		slice           *resourcev1.ResourceSlice
		requestDriver   string        // defaults to the slice's driver
		age             time.Duration // since creation
		updateStatusErr error
		failListing     bool
		wantErr         error  // nil, errIncompletePool, or errAny
		wantLabel       string // driver_name; defaults to the request's driver
		wantResult      string // empty if nothing was processed
		wantErrors      float64
		wantDurations   uint64
		wantEventReason string
	}{
		"complete pool": {
			slice:         makeSlice("slice-1", "complete.example.com", "pool-1", "node-1", 4),
			wantResult:    metrics.ResultComplete,
			wantDurations: 1,
		},
		"driver without slices is labeled unknown": {
			slice:         makeSlice("slice-1", "known.example.com", "pool-1", "node-1", 4),
			requestDriver: "typo.example.com",
			wantLabel:     metrics.UnknownDriver,
			wantResult:    metrics.ResultComplete,
			wantDurations: 1,
		},
		"incomplete pool within the wait window is requeued silently": {
			slice:   makeSliceWithExpectedCount("slice-1", "requeue.example.com", "pool-1", "node-1", 4, 1, 3),
			wantErr: errIncompletePool,
		},
		"incomplete pool after the wait window is written and reported": {
			slice:           makeSliceWithExpectedCount("slice-1", "giveup.example.com", "pool-1", "node-1", 4, 1, 3),
			age:             20 * time.Second,
			wantResult:      metrics.ResultIncomplete,
			wantDurations:   1,
			wantEventReason: eventReasonPoolIncomplete,
		},
		// Retried quietly, not an error.
		"status update conflict": {
			slice:           makeSlice("slice-1", "conflict.example.com", "pool-1", "node-1", 4),
			updateStatusErr: conflict,
			wantErr:         errStatusConflict,
		},
		"status update failure": {
			slice:           makeSlice("slice-1", "updatefail.example.com", "pool-1", "node-1", 4),
			updateStatusErr: errors.New("injected update failure"),
			wantErr:         errAny,
			wantResult:      metrics.ResultError,
			wantErrors:      1,
			wantDurations:   1,
		},
		"calculation failure": {
			slice:           makeSlice("slice-1", "listfail.example.com", "pool-1", "node-1", 4),
			failListing:     true,
			wantLabel:       metrics.UnknownDriver,
			wantResult:      metrics.ResultFailed,
			wantDurations:   1,
			wantEventReason: eventReasonCalculationFailed,
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
			request.CreationTimestamp = metav1.NewTime(time.Now().Add(-tc.age))

			fakeClient := fake.NewClientset(request)
			if tc.updateStatusErr != nil {
				fakeClient.PrependReactor("update", "resourcepoolstatusrequests", func(action k8stesting.Action) (bool, runtime.Object, error) {
					return action.GetSubresource() == "status", nil, tc.updateStatusErr
				})
			}
			controller, informerFactory := newObservabilityController(t, ctx, fakeClient)
			recorder := record.NewFakeRecorder(10)
			controller.recorder = recorder
			addToStore(t, informerFactory.Resource().V1alpha3().ResourcePoolStatusRequests().Informer().GetStore(), request)
			addToStore(t, informerFactory.Resource().V1().ResourceSlices().Informer().GetStore(), tc.slice)
			if tc.failListing {
				controller.sliceIndexer = failingSliceIndexer{controller.sliceIndexer}
			}

			durationSum, err := testutil.GetHistogramMetricValue(metrics.RequestProcessingDuration.WithLabelValues(label))
			if err != nil {
				t.Fatalf("read histogram: %v", err)
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
			if got := durations(); got != tc.wantDurations {
				t.Errorf("request_processing_duration_seconds count delta = %v, want %v", got, tc.wantDurations)
			}
			// The duration covers the sync only, not the time since creation.
			if tc.age > 0 && tc.wantDurations > 0 {
				after, err := testutil.GetHistogramMetricValue(metrics.RequestProcessingDuration.WithLabelValues(label))
				if err != nil {
					t.Fatalf("read histogram: %v", err)
				}
				if got := after - durationSum; got >= tc.age.Seconds() {
					t.Errorf("Observed duration %vs includes the %v before the sync", got, tc.age)
				}
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

// Incomplete-pool requeues must not use the retry budget, so a conflict on the
// final write is still retried instead of dropping the request.
func TestIncompleteRequeueDoesNotUseRetryBudget(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)
	slice := makeSliceWithExpectedCount("slice-1", "budget.example.com", "pool-1", "node-1", 4, 1, 3)
	request := makeRequest(slice.Spec.Driver)
	request.CreationTimestamp = metav1.Now()

	fakeClient := fake.NewClientset(request)
	conflicts := maxRetries
	fakeClient.PrependReactor("update", "resourcepoolstatusrequests", func(action k8stesting.Action) (bool, runtime.Object, error) {
		if action.GetSubresource() != "status" || conflicts == 0 {
			return false, nil, nil
		}
		conflicts--
		return true, nil, apierrors.NewConflict(schema.GroupResource{Resource: "resourcepoolstatusrequests"}, request.Name, errors.New("modified"))
	})
	controller, informerFactory := newObservabilityController(t, ctx, fakeClient)
	defer controller.workqueue.ShutDown()
	requests := informerFactory.Resource().V1alpha3().ResourcePoolStatusRequests().Informer().GetStore()
	addToStore(t, requests, request)
	addToStore(t, informerFactory.Resource().V1().ResourceSlices().Informer().GetStore(), slice)

	controller.workqueue.Add(request.Name)
	controller.processNextWorkItem(ctx)
	if got := controller.workqueue.NumRequeues(request.Name); got != 0 {
		t.Fatalf("NumRequeues = %d after an incomplete-pool requeue, want 0", got)
	}

	// Past the window, maxRetries conflicts in a row are still retried.
	request.CreationTimestamp = metav1.NewTime(time.Now().Add(-incompleteWaitWindow))
	if err := requests.Update(request); err != nil {
		t.Fatalf("Failed to update request in informer: %v", err)
	}
	for range maxRetries + 1 {
		// A dropped key would block the next Get forever.
		if err := wait.PollUntilContextTimeout(ctx, 10*time.Millisecond, 5*time.Second, true, func(context.Context) (bool, error) {
			return controller.workqueue.Len() > 0, nil
		}); err != nil {
			t.Fatal("Request was dropped instead of being retried")
		}
		controller.processNextWorkItem(ctx)
	}
	updated, err := fakeClient.ResourceV1alpha3().ResourcePoolStatusRequests().Get(ctx, request.Name, metav1.GetOptions{})
	if err != nil {
		t.Fatalf("Failed to get request: %v", err)
	}
	if updated.Status == nil {
		t.Fatal("Expected the partial status to be written after the conflicts, got none")
	}
}

// A request recreated with the same name must not inherit the retry count.
func TestDeleteForgetsRetries(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)
	request := makeRequest("forget.example.com")
	fakeClient := fake.NewClientset()
	controller, _ := newObservabilityController(t, ctx, fakeClient)
	defer controller.workqueue.ShutDown()

	for range 3 {
		controller.workqueue.AddRateLimited(request.Name)
	}
	controller.forgetRequest(cache.DeletedFinalStateUnknown{Key: request.Name, Obj: request})
	if got := controller.workqueue.NumRequeues(request.Name); got != 0 {
		t.Errorf("NumRequeues = %d after delete, want 0", got)
	}
}

func newObservabilityController(t *testing.T, ctx context.Context, client *fake.Clientset) (*Controller, informers.SharedInformerFactory) {
	t.Helper()
	informerFactory := informers.NewSharedInformerFactory(client, 0)
	controller, err := NewController(ctx, client,
		informerFactory.Resource().V1alpha3().ResourcePoolStatusRequests(),
		informerFactory.Resource().V1().ResourceSlices(),
		informerFactory.Resource().V1().ResourceClaims(),
		informerFactory.Resource().V1().DeviceTaintRules(),
	)
	if err != nil {
		t.Fatalf("Failed to create controller: %v", err)
	}
	return controller, informerFactory
}

func addToStore(t *testing.T, store cache.Store, obj interface{}) {
	t.Helper()
	if err := store.Add(obj); err != nil {
		t.Fatalf("Failed to add %T to informer: %v", obj, err)
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
