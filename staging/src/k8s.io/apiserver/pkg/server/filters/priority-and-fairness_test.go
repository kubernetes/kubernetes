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
	"fmt"
	"net/http"
	"net/http/httptest"
	"net/url"
	"reflect"
	"strings"
	"sync"
	"testing"
	"time"

	flowcontrol "k8s.io/api/flowcontrol/v1beta1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apiserver/pkg/apis/flowcontrol/bootstrap"
	"k8s.io/apiserver/pkg/authentication/user"
	apifilters "k8s.io/apiserver/pkg/endpoints/filters"
	epmetrics "k8s.io/apiserver/pkg/endpoints/metrics"
	apirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/server/mux"
	utilflowcontrol "k8s.io/apiserver/pkg/util/flowcontrol"
	fq "k8s.io/apiserver/pkg/util/flowcontrol/fairqueuing"
	fcmetrics "k8s.io/apiserver/pkg/util/flowcontrol/metrics"
	"k8s.io/client-go/informers"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/kubernetes/fake"
	"k8s.io/component-base/metrics/legacyregistry"
	"k8s.io/component-base/metrics/testutil"
)

type mockDecision int

const (
	decisionNoQueuingExecute mockDecision = iota
	decisionQueuingExecute
	decisionCancelWait
	decisionReject
	decisionSkipFilter
)

type fakeApfFilter struct {
	mockDecision mockDecision
	postEnqueue  func()
	postDequeue  func()
}

func (t fakeApfFilter) MaintainObservations(stopCh <-chan struct{}) {
}

func (t fakeApfFilter) Handle(ctx context.Context,
	requestDigest utilflowcontrol.RequestDigest,
	noteFn func(fs *flowcontrol.FlowSchema, pl *flowcontrol.PriorityLevelConfiguration),
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

func newApfServerWithSingleRequest(decision mockDecision, t *testing.T) *httptest.Server {
	onExecuteFunc := func() {
		if decision == decisionCancelWait {
			t.Errorf("execute should not be invoked")
		}
		// atomicReadOnlyExecuting can be either 0 or 1 as we test one request at a time.
		if decision != decisionSkipFilter && atomicReadOnlyExecuting != 1 {
			t.Errorf("Wanted %d requests executing, got %d", 1, atomicReadOnlyExecuting)
		}
	}
	postExecuteFunc := func() {}
	// atomicReadOnlyWaiting can be either 0 or 1 as we test one request at a time.
	postEnqueueFunc := func() {
		if atomicReadOnlyWaiting != 1 {
			t.Errorf("Wanted %d requests in queue, got %d", 1, atomicReadOnlyWaiting)
		}
	}
	postDequeueFunc := func() {
		if atomicReadOnlyWaiting != 0 {
			t.Errorf("Wanted %d requests in queue, got %d", 0, atomicReadOnlyWaiting)
		}
	}
	return newApfServerWithHooks(decision, onExecuteFunc, postExecuteFunc, postEnqueueFunc, postDequeueFunc, t)
}

func newApfServerWithHooks(decision mockDecision, onExecute, postExecute, postEnqueue, postDequeue func(), t *testing.T) *httptest.Server {
	requestInfoFactory := &apirequest.RequestInfoFactory{APIPrefixes: sets.NewString("apis", "api"), GrouplessAPIPrefixes: sets.NewString("api")}
	longRunningRequestCheck := BasicLongRunningRequestCheck(sets.NewString("watch"), sets.NewString("proxy"))

	apfHandler := WithPriorityAndFairness(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		onExecute()
	}), longRunningRequestCheck, fakeApfFilter{
		mockDecision: decision,
		postEnqueue:  postEnqueue,
		postDequeue:  postDequeue,
	})

	handler := apifilters.WithRequestInfo(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		r = r.WithContext(apirequest.WithUser(r.Context(), &user.DefaultInfo{
			Groups: []string{user.AllUnauthenticated},
		}))
		apfHandler.ServeHTTP(w, r)
		postExecute()
		if atomicReadOnlyExecuting != 0 {
			t.Errorf("Wanted %d requests executing, got %d", 0, atomicReadOnlyExecuting)
		}
	}), requestInfoFactory)

	apfServer := httptest.NewServer(handler)
	return apfServer
}

func TestApfSkipLongRunningRequest(t *testing.T) {
	epmetrics.Register()

	server := newApfServerWithSingleRequest(decisionSkipFilter, t)
	defer server.Close()

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	StartPriorityAndFairnessWatermarkMaintenance(ctx.Done())

	// send a watch request to test skipping long running request
	if err := expectHTTPGet(fmt.Sprintf("%s/api/v1/namespaces?watch=true", server.URL), http.StatusOK); err != nil {
		// request should not be rejected
		t.Error(err)
	}
}

func TestApfRejectRequest(t *testing.T) {
	epmetrics.Register()

	server := newApfServerWithSingleRequest(decisionReject, t)
	defer server.Close()

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	StartPriorityAndFairnessWatermarkMaintenance(ctx.Done())

	if err := expectHTTPGet(fmt.Sprintf("%s/api/v1/namespaces/default", server.URL), http.StatusTooManyRequests); err != nil {
		t.Error(err)
	}

	checkForExpectedMetrics(t, []string{
		"apiserver_request_terminations_total",
		"apiserver_dropped_requests_total",
	})
}

func TestApfExemptRequest(t *testing.T) {
	epmetrics.Register()
	fcmetrics.Register()

	// Wait for at least one sampling window to pass since creation of metrics.ReadWriteConcurrencyObserverPairGenerator,
	// so that an observation will cause some data to go into the Prometheus metrics.
	time.Sleep(time.Millisecond * 50)

	server := newApfServerWithSingleRequest(decisionNoQueuingExecute, t)
	defer server.Close()

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	StartPriorityAndFairnessWatermarkMaintenance(ctx.Done())

	if err := expectHTTPGet(fmt.Sprintf("%s/api/v1/namespaces/default", server.URL), http.StatusOK); err != nil {
		t.Error(err)
	}

	checkForExpectedMetrics(t, []string{
		"apiserver_current_inflight_requests",
		"apiserver_flowcontrol_read_vs_write_request_count_watermarks",
		"apiserver_flowcontrol_read_vs_write_request_count_samples",
	})
}

func TestApfExecuteRequest(t *testing.T) {
	epmetrics.Register()
	fcmetrics.Register()

	// Wait for at least one sampling window to pass since creation of metrics.ReadWriteConcurrencyObserverPairGenerator,
	// so that an observation will cause some data to go into the Prometheus metrics.
	time.Sleep(time.Millisecond * 50)

	server := newApfServerWithSingleRequest(decisionQueuingExecute, t)
	defer server.Close()

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	StartPriorityAndFairnessWatermarkMaintenance(ctx.Done())

	if err := expectHTTPGet(fmt.Sprintf("%s/api/v1/namespaces/default", server.URL), http.StatusOK); err != nil {
		t.Error(err)
	}

	checkForExpectedMetrics(t, []string{
		"apiserver_current_inflight_requests",
		"apiserver_current_inqueue_requests",
		"apiserver_flowcontrol_read_vs_write_request_count_watermarks",
		"apiserver_flowcontrol_read_vs_write_request_count_samples",
	})
}

func TestApfExecuteMultipleRequests(t *testing.T) {
	epmetrics.Register()
	fcmetrics.Register()

	// Wait for at least one sampling window to pass since creation of metrics.ReadWriteConcurrencyObserverPairGenerator,
	// so that an observation will cause some data to go into the Prometheus metrics.
	time.Sleep(time.Millisecond * 50)

	concurrentRequests := 5
	preStartExecute, postStartExecute := &sync.WaitGroup{}, &sync.WaitGroup{}
	preEnqueue, postEnqueue := &sync.WaitGroup{}, &sync.WaitGroup{}
	preDequeue, postDequeue := &sync.WaitGroup{}, &sync.WaitGroup{}
	finishExecute := &sync.WaitGroup{}
	for _, wg := range []*sync.WaitGroup{preStartExecute, postStartExecute, preEnqueue, postEnqueue, preDequeue, postDequeue, finishExecute} {
		wg.Add(concurrentRequests)
	}

	onExecuteFunc := func() {
		preStartExecute.Done()
		preStartExecute.Wait()
		if int(atomicReadOnlyExecuting) != concurrentRequests {
			t.Errorf("Wanted %d requests executing, got %d", concurrentRequests, atomicReadOnlyExecuting)
		}
		postStartExecute.Done()
		postStartExecute.Wait()
	}

	postEnqueueFunc := func() {
		preEnqueue.Done()
		preEnqueue.Wait()
		if int(atomicReadOnlyWaiting) != concurrentRequests {
			t.Errorf("Wanted %d requests in queue, got %d", 1, atomicReadOnlyWaiting)

		}
		postEnqueue.Done()
		postEnqueue.Wait()
	}

	postDequeueFunc := func() {
		preDequeue.Done()
		preDequeue.Wait()
		if atomicReadOnlyWaiting != 0 {
			t.Errorf("Wanted %d requests in queue, got %d", 0, atomicReadOnlyWaiting)
		}
		postDequeue.Done()
		postDequeue.Wait()
	}

	postExecuteFunc := func() {
		finishExecute.Done()
		finishExecute.Wait()
	}

	server := newApfServerWithHooks(decisionQueuingExecute, onExecuteFunc, postExecuteFunc, postEnqueueFunc, postDequeueFunc, t)
	defer server.Close()

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	StartPriorityAndFairnessWatermarkMaintenance(ctx.Done())

	var wg sync.WaitGroup
	wg.Add(concurrentRequests)
	for i := 0; i < concurrentRequests; i++ {
		go func() {
			defer wg.Done()
			if err := expectHTTPGet(fmt.Sprintf("%s/api/v1/namespaces/default", server.URL), http.StatusOK); err != nil {
				t.Error(err)
			}
		}()
	}
	wg.Wait()

	checkForExpectedMetrics(t, []string{
		"apiserver_current_inflight_requests",
		"apiserver_current_inqueue_requests",
		"apiserver_flowcontrol_read_vs_write_request_count_watermarks",
		"apiserver_flowcontrol_read_vs_write_request_count_samples",
	})
}

func TestApfCancelWaitRequest(t *testing.T) {
	epmetrics.Register()

	server := newApfServerWithSingleRequest(decisionCancelWait, t)
	defer server.Close()

	if err := expectHTTPGet(fmt.Sprintf("%s/api/v1/namespaces/default", server.URL), http.StatusTooManyRequests); err != nil {
		t.Error(err)
	}

	checkForExpectedMetrics(t, []string{
		"apiserver_current_inflight_requests",
		"apiserver_request_terminations_total",
		"apiserver_dropped_requests_total",
	})
}

func TestPriorityAndFairnessWithPanicRecoverAndTimeoutFilter(t *testing.T) {
	fcmetrics.Register()

	t.Run("priority level concurrency is set to 1, request handler panics, next request should not be rejected", func(t *testing.T) {
		const (
			requestTimeout                                        = time.Minute
			userName                                              = "alice"
			fsName                                                = "test-fs"
			plName                                                = "test-pl"
			serverConcurrency, plConcurrencyShares, plConcurrency = 1, 1, 1
		)

		objects := newConfiguration(fsName, plName, userName, flowcontrol.LimitResponseTypeReject, plConcurrencyShares)
		clientset := newClientset(t, objects...)
		// this test does not rely on resync, so resync period is set to zero
		factory := informers.NewSharedInformerFactory(clientset, 0)
		controller := utilflowcontrol.New(factory, clientset.FlowcontrolV1beta1(), serverConcurrency, requestTimeout/4)

		stopCh, controllerCompletedCh := make(chan struct{}), make(chan struct{})
		factory.Start(stopCh)

		// wait for the informer cache to sync.
		timeout, cancel := context.WithTimeout(context.TODO(), 5*time.Second)
		defer cancel()
		cacheSyncDone := factory.WaitForCacheSync(timeout.Done())
		if names := unsyncedInformers(cacheSyncDone); len(names) > 0 {
			t.Fatalf("WaitForCacheSync did not successfully complete, resources=%#v", names)
		}

		var controllerErr error
		go func() {
			defer close(controllerCompletedCh)
			controllerErr = controller.Run(stopCh)
		}()

		// make sure that apf controller syncs the priority level configuration object we are using in this test.
		// read the metrics and ensure the concurrency limit for our priority level is set to the expected value.
		pollErr := wait.PollImmediate(100*time.Millisecond, 5*time.Second, func() (done bool, err error) {
			if err := gaugeValueMatch("apiserver_flowcontrol_request_concurrency_limit", map[string]string{"priority_level": plName}, plConcurrency); err != nil {
				t.Logf("polling retry - error: %s", err)
				return false, nil
			}
			return true, nil
		})
		if pollErr != nil {
			t.Fatalf("expected the apf controller to sync the priotity level configuration object: %s", "test-pl")
		}

		var executed bool
		// we will raise a panic for the first request.
		firstRequestPathPanic := "/request/panic"
		requestHandler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			executed = true
			expectMatchingAPFHeaders(t, w, fsName, plName)

			if r.URL.Path == firstRequestPathPanic {
				panic(fmt.Errorf("request handler panic'd as designed - %#v", r.RequestURI))
			}
		})
		handler := newHandlerChain(t, requestHandler, controller, userName, requestTimeout)

		server, requestGetter := newHTTP2ServerWithClient(handler)
		defer server.Close()

		var err error
		_, err = requestGetter(firstRequestPathPanic)
		if !executed {
			t.Errorf("expected inner handler to be executed for request: %s", firstRequestPathPanic)
		}
		expectResetStreamError(t, err)

		executed = false
		// the second request should be served successfully.
		secondRequestPathShouldWork := "/request/should-work"
		response, err := requestGetter(secondRequestPathShouldWork)
		if !executed {
			t.Errorf("expected inner handler to be executed for request: %s", secondRequestPathShouldWork)
		}
		if err != nil {
			t.Errorf("expected request: %s to succeed, but got error: %#v", secondRequestPathShouldWork, err)
		}
		if response.StatusCode != http.StatusOK {
			t.Errorf("expected HTTP status code: %d for request: %s, but got: %#v", http.StatusOK, secondRequestPathShouldWork, response)
		}

		close(stopCh)
		t.Log("waiting for the controller to shutdown")
		<-controllerCompletedCh

		if controllerErr != nil {
			t.Errorf("expected a nil error from controller, but got: %#v", controllerErr)
		}
	})
}

// returns a started http2 server, with a client function to send request to the server.
func newHTTP2ServerWithClient(handler http.Handler) (*httptest.Server, func(path string) (*http.Response, error)) {
	server := httptest.NewUnstartedServer(handler)
	server.EnableHTTP2 = true
	server.StartTLS()

	return server, func(path string) (*http.Response, error) {
		return server.Client().Get(server.URL + path)
	}
}

// verifies that the expected flow schema and priority level UIDs are attached to the header.
func expectMatchingAPFHeaders(t *testing.T, w http.ResponseWriter, expectedFS, expectedPL string) {
	if w == nil {
		t.Fatal("expected a non nil HTTP response")
	}

	key := flowcontrol.ResponseHeaderMatchedFlowSchemaUID
	if value := w.Header().Get(key); expectedFS != value {
		t.Fatalf("expected HTTP header %s to have value %q, but got: %q", key, expectedFS, value)
	}

	key = flowcontrol.ResponseHeaderMatchedPriorityLevelConfigurationUID
	if value := w.Header().Get(key); expectedPL != value {
		t.Fatalf("expected HTTP header %s to have value %q, but got %q", key, expectedPL, value)
	}
}

// when a request panics, http2 resets the stream with an INTERNAL_ERROR message
func expectResetStreamError(t *testing.T, err error) {
	if err == nil {
		t.Fatalf("expected the server to send an error, but got nil")
	}

	uerr, ok := err.(*url.Error)
	if !ok {
		t.Fatalf("expected the error to be of type *url.Error, but got: %T", err)
	}
	if !strings.Contains(uerr.Error(), "INTERNAL_ERROR") {
		t.Fatalf("expected a stream reset error, but got: %s", uerr.Error())
	}
}

func newClientset(t *testing.T, objects ...runtime.Object) clientset.Interface {
	clientset := fake.NewSimpleClientset(objects...)
	if clientset == nil {
		t.Fatal("unable to create fake client set")
	}
	return clientset
}

// builds a chain of handlers that include the panic recovery and timeout filter, so we can simulate the behavior of
// a real apiserver.
// the specified user is added as the authenticated user to the request context.
func newHandlerChain(t *testing.T, handler http.Handler, filter utilflowcontrol.Interface, userName string, requestTimeout time.Duration) http.Handler {
	requestInfoFactory := &apirequest.RequestInfoFactory{APIPrefixes: sets.NewString("apis", "api"), GrouplessAPIPrefixes: sets.NewString("api")}
	longRunningRequestCheck := BasicLongRunningRequestCheck(sets.NewString("watch"), sets.NewString("proxy"))

	apfHandler := WithPriorityAndFairness(handler, longRunningRequestCheck, filter)

	// add the handler in the chain that adds the specified user to the request context
	handler = http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		r = r.WithContext(apirequest.WithUser(r.Context(), &user.DefaultInfo{
			Name:   userName,
			Groups: []string{user.AllAuthenticated},
		}))

		apfHandler.ServeHTTP(w, r)
	})

	handler = WithTimeoutForNonLongRunningRequests(handler, longRunningRequestCheck, requestTimeout)
	handler = apifilters.WithRequestInfo(handler, requestInfoFactory)
	handler = WithPanicRecovery(handler, requestInfoFactory)
	return handler
}

func unsyncedInformers(status map[reflect.Type]bool) []string {
	names := make([]string, 0)

	for objType, synced := range status {
		if !synced {
			names = append(names, objType.Name())
		}
	}

	return names
}

func newConfiguration(fsName, plName, user string, responseType flowcontrol.LimitResponseType, concurrency int32) []runtime.Object {
	fs := &flowcontrol.FlowSchema{
		ObjectMeta: metav1.ObjectMeta{
			Name: fsName,
			UID:  types.UID(fsName),
		},
		Spec: flowcontrol.FlowSchemaSpec{
			MatchingPrecedence: 1,
			PriorityLevelConfiguration: flowcontrol.PriorityLevelConfigurationReference{
				Name: plName,
			},
			DistinguisherMethod: &flowcontrol.FlowDistinguisherMethod{
				Type: flowcontrol.FlowDistinguisherMethodByUserType,
			},
			Rules: []flowcontrol.PolicyRulesWithSubjects{
				{
					Subjects: []flowcontrol.Subject{
						{
							Kind: flowcontrol.SubjectKindUser,
							User: &flowcontrol.UserSubject{
								Name: user,
							},
						},
					},
					NonResourceRules: []flowcontrol.NonResourcePolicyRule{
						{
							Verbs:           []string{flowcontrol.VerbAll},
							NonResourceURLs: []string{flowcontrol.NonResourceAll},
						},
					},
				},
			},
		},
	}

	pl := &flowcontrol.PriorityLevelConfiguration{
		ObjectMeta: metav1.ObjectMeta{
			Name: plName,
			UID:  types.UID(plName),
		},
		Spec: flowcontrol.PriorityLevelConfigurationSpec{
			Type: flowcontrol.PriorityLevelEnablementLimited,
			Limited: &flowcontrol.LimitedPriorityLevelConfiguration{
				AssuredConcurrencyShares: concurrency,
				LimitResponse: flowcontrol.LimitResponse{
					Type: responseType,
				},
			},
		},
	}

	return []runtime.Object{fs, pl}
}

// gathers and checks the metrics.
func checkForExpectedMetrics(t *testing.T, expectedMetrics []string) {
	metricsFamily, err := legacyregistry.DefaultGatherer.Gather()
	if err != nil {
		t.Fatalf("Failed to gather metrics %v", err)
	}

	metrics := map[string]interface{}{}
	for _, mf := range metricsFamily {
		metrics[*mf.Name] = mf
	}

	for _, metricName := range expectedMetrics {
		if _, ok := metrics[metricName]; !ok {
			if !ok {
				t.Errorf("Scraped metrics did not include expected metric %s", metricName)
			}
		}
	}
}

// gaugeValueMatch ensures that the value of gauge metrics matching the labelFilter is as expected.
func gaugeValueMatch(name string, labelFilter map[string]string, wantValue int) error {
	metrics, err := legacyregistry.DefaultGatherer.Gather()
	if err != nil {
		return fmt.Errorf("failed to gather metrics: %s", err)
	}

	sum := 0
	familyMatch, labelMatch := false, false
	for _, mf := range metrics {
		if mf.GetName() != name {
			continue
		}

		familyMatch = true
		for _, metric := range mf.GetMetric() {
			if !testutil.LabelsMatch(metric, labelFilter) {
				continue
			}

			labelMatch = true
			sum += int(metric.GetGauge().GetValue())
		}
	}
	if !familyMatch {
		return fmt.Errorf("expected to find the metric family: %s in the gathered result", name)
	}
	if !labelMatch {
		return fmt.Errorf("expected to find metrics with matching labels: %#+v", labelFilter)
	}
	if wantValue != sum {
		return fmt.Errorf("expected the sum to be: %d, but got: %d for gauge metric: %s with labels %#+v", wantValue, sum, name, labelFilter)
	}

	return nil
}
