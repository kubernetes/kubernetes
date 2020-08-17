/*
Copyright 2016 The Kubernetes Authors.

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

package filters

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	fctypesv1a1 "k8s.io/api/flowcontrol/v1alpha1"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apiserver/pkg/apis/flowcontrol/bootstrap"
	"k8s.io/apiserver/pkg/authentication/user"
	apifilters "k8s.io/apiserver/pkg/endpoints/filters"
	epmetrics "k8s.io/apiserver/pkg/endpoints/metrics"
	apirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/server/mux"
	utilflowcontrol "k8s.io/apiserver/pkg/util/flowcontrol"
	fq "k8s.io/apiserver/pkg/util/flowcontrol/fairqueuing"
	fcmetrics "k8s.io/apiserver/pkg/util/flowcontrol/metrics"
	"k8s.io/component-base/metrics/legacyregistry"
)

const (
	decisionNoQueuingExecute = iota
	decisionQueuingExecute
	decisionCancelWait
	decisionReject
	decisionSkipFilter
)

type fakeApfFilter struct {
	mockDecision int
	postEnqueue  func()
	postDequeue  func()
}

func (t fakeApfFilter) MaintainObservations(stopCh <-chan struct{}) {
}

func (t fakeApfFilter) Handle(ctx context.Context,
	requestDigest utilflowcontrol.RequestDigest,
	noteFn func(fs *fctypesv1a1.FlowSchema, pl *fctypesv1a1.PriorityLevelConfiguration),
	queueNoteFn fq.QueueNoteFn,
	execFn func(),
) {
	if t.mockDecision == decisionSkipFilter {
		panic("Handle should not be invoked")
	}
	noteFn(bootstrap.SuggestedFlowSchemaGlobalDefault, bootstrap.SuggestedPriorityLevelConfigurationGlobalDefault)
	switch t.mockDecision {
	case decisionNoQueuingExecute:
		execFn()
	case decisionQueuingExecute:
		queueNoteFn(true)
		t.postEnqueue()
		queueNoteFn(false)
		t.postDequeue()
		execFn()
	case decisionCancelWait:
		queueNoteFn(true)
		t.postEnqueue()
		queueNoteFn(false)
		t.postDequeue()
	case decisionReject:
		return
	}
}

func (t fakeApfFilter) Run(stopCh <-chan struct{}) error {
	return nil
}

func (t fakeApfFilter) Install(c *mux.PathRecorderMux) {
}

func newApfServer(decision int, t *testing.T) *httptest.Server {
	requestInfoFactory := &apirequest.RequestInfoFactory{APIPrefixes: sets.NewString("apis", "api"), GrouplessAPIPrefixes: sets.NewString("api")}
	longRunningRequestCheck := BasicLongRunningRequestCheck(sets.NewString("watch"), sets.NewString("proxy"))

	apfHandler := WithPriorityAndFairness(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if decision == decisionCancelWait {
			t.Errorf("execute should not be invoked")
		}
		if decision != decisionSkipFilter && atomicReadOnlyExecuting != 1 {
			t.Errorf("Wanted %d requests executing, got %d", 1, atomicReadOnlyExecuting)
		}
	}), longRunningRequestCheck, fakeApfFilter{
		mockDecision: decision,
		postEnqueue: func() {
			if atomicReadOnlyWaiting != 1 {
				t.Errorf("Wanted %d requests in queue, got %d", 1, atomicReadOnlyWaiting)
			}
		},
		postDequeue: func() {
			if atomicReadOnlyWaiting != 0 {
				t.Errorf("Wanted %d requests in queue, got %d", 0, atomicReadOnlyWaiting)
			}
		},
	})

	handler := apifilters.WithRequestInfo(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		r = r.WithContext(apirequest.WithUser(r.Context(), &user.DefaultInfo{
			Groups: []string{user.AllUnauthenticated},
		}))
		apfHandler.ServeHTTP(w, r)
		if atomicReadOnlyExecuting != 0 {
			t.Errorf("Wanted %d requests executing, got %d", 0, atomicReadOnlyExecuting)
		}
	}), requestInfoFactory)

	apfServer := httptest.NewServer(handler)
	return apfServer
}

func TestApfSkipLongRunningRequest(t *testing.T) {
	epmetrics.Register()

	server := newApfServer(decisionSkipFilter, t)
	defer server.Close()

	if err := expectHTTPGet(fmt.Sprintf("%s/api/v1/namespaces?watch=true", server.URL), http.StatusOK); err != nil {
		// request should not be rejected
		t.Error(err)
	}
}

func TestApfRejectRequest(t *testing.T) {
	epmetrics.Register()

	server := newApfServer(decisionReject, t)
	defer server.Close()

	if err := expectHTTPGet(fmt.Sprintf("%s/api/v1/namespaces/default", server.URL), http.StatusTooManyRequests); err != nil {
		t.Error(err)
	}

	checkForExpectedMetricsWithRetry(t, []string{
		"apiserver_request_terminations_total",
		"apiserver_dropped_requests_total",
	})
}

func TestApfExemptRequest(t *testing.T) {
	epmetrics.Register()
	fcmetrics.Register()

	// wait the first sampleAndWaterMark metrics to be collected
	time.Sleep(time.Millisecond * 50)

	server := newApfServer(decisionNoQueuingExecute, t)
	defer server.Close()

	if err := expectHTTPGet(fmt.Sprintf("%s/api/v1/namespaces/default", server.URL), http.StatusOK); err != nil {
		t.Error(err)
	}

	checkForExpectedMetricsWithRetry(t, []string{
		"apiserver_current_inflight_requests",
		"apiserver_flowcontrol_read_vs_write_request_count_watermarks",
		"apiserver_flowcontrol_read_vs_write_request_count_samples",
	})
}

func TestApfExecuteRequest(t *testing.T) {
	epmetrics.Register()
	fcmetrics.Register()

	// wait the first sampleAndWaterMark metrics to be collected
	time.Sleep(time.Millisecond * 50)

	server := newApfServer(decisionQueuingExecute, t)
	defer server.Close()

	if err := expectHTTPGet(fmt.Sprintf("%s/api/v1/namespaces/default", server.URL), http.StatusOK); err != nil {
		t.Error(err)
	}

	checkForExpectedMetricsWithRetry(t, []string{
		"apiserver_current_inflight_requests",
		"apiserver_current_inqueue_requests",
		"apiserver_flowcontrol_read_vs_write_request_count_watermarks",
		"apiserver_flowcontrol_read_vs_write_request_count_samples",
	})
}

func TestApfCancelWaitRequest(t *testing.T) {
	epmetrics.Register()

	server := newApfServer(decisionCancelWait, t)
	defer server.Close()

	if err := expectHTTPGet(fmt.Sprintf("%s/api/v1/namespaces/default", server.URL), http.StatusTooManyRequests); err != nil {
		t.Error(err)
	}

	checkForExpectedMetricsWithRetry(t, []string{
		"apiserver_current_inflight_requests",
		"apiserver_request_terminations_total",
		"apiserver_dropped_requests_total",
	})
}

// wait async metrics to be collected
func checkForExpectedMetricsWithRetry(t *testing.T, expectedMetrics []string) {
	maxRetries := 5
	var checkErrors []error
	for i := 0; i < maxRetries; i++ {
		t.Logf("Check for expected metrics with retry %d", i)
		metricsFamily, err := legacyregistry.DefaultGatherer.Gather()
		if err != nil {
			t.Fatalf("Failed to gather metrics %v", err)
		}

		metrics := map[string]interface{}{}
		for _, mf := range metricsFamily {
			mf := mf
			metrics[*mf.Name] = mf
		}

		checkErrors = checkForExpectedMetrics(expectedMetrics, metrics)
		if checkErrors == nil {
			return
		}

		time.Sleep(1 * time.Second)
	}
	for _, checkError := range checkErrors {
		t.Error(checkError)
	}
}

func checkForExpectedMetrics(expectedMetrics []string, metrics map[string]interface{}) []error {
	var errs []error
	for _, metricName := range expectedMetrics {
		if _, ok := metrics[metricName]; !ok {
			if !ok {
				errs = append(errs, errors.New("Scraped metrics did not include expected metric "+metricName))
			}
		}
	}
	return errs
}
