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

package server

import (
	"context"
	"crypto/tls"
	"crypto/x509"
	"fmt"
	"io/ioutil"
	"log"
	"net"
	"net/http"
	"net/http/httptrace"
	"reflect"
	"sync"
	"testing"
	"time"

	"k8s.io/apimachinery/pkg/util/wait"
	genericapifilters "k8s.io/apiserver/pkg/endpoints/filters"
	"k8s.io/apiserver/pkg/server/dynamiccertificates"
	genericfilters "k8s.io/apiserver/pkg/server/filters"

	"github.com/google/go-cmp/cmp"
	"golang.org/x/net/http2"
)

// doer sends a request to the server
type doer func(client *http.Client, gci func(httptrace.GotConnInfo), path string, timeout time.Duration) result

func (d doer) Do(client *http.Client, gci func(httptrace.GotConnInfo), path string, timeout time.Duration) result {
	return d(client, gci, path, timeout)
}

type result struct {
	err      error
	response *http.Response
}

// wrap a lifecycleSignal so the test can inject its own callback
type wrappedLifecycleSignal struct {
	lifecycleSignal
	callback func(bool, string, lifecycleSignal)
}

func (w *wrappedLifecycleSignal) Signal() {
	var name string
	if ncw, ok := w.lifecycleSignal.(*namedChannelWrapper); ok {
		name = ncw.name
	}

	// the callback is invoked before and after the termination event is signaled
	if w.callback != nil {
		w.callback(true, name, w.lifecycleSignal)
	}
	w.lifecycleSignal.Signal()
	if w.callback != nil {
		w.callback(false, name, w.lifecycleSignal)
	}
}

func wrapLifecycleSignals(t *testing.T, ts *lifecycleSignals, callback func(bool, string, lifecycleSignal)) {
	newWrappedLifecycleSignal := func(delegated lifecycleSignal) lifecycleSignal {
		return &wrappedLifecycleSignal{
			lifecycleSignal: delegated,
			callback:        callback,
		}
	}

	ts.AfterShutdownDelayDuration = newWrappedLifecycleSignal(ts.AfterShutdownDelayDuration)
	ts.HTTPServerStoppedListening = newWrappedLifecycleSignal(ts.HTTPServerStoppedListening)
	ts.InFlightRequestsDrained = newWrappedLifecycleSignal(ts.InFlightRequestsDrained)
	ts.ShutdownInitiated = newWrappedLifecycleSignal(ts.ShutdownInitiated)
}

type step struct {
	waitCh, doneCh chan struct{}
	fn             func()
}

func (s step) done() <-chan struct{} {
	close(s.waitCh)
	return s.doneCh
}
func (s step) execute() {
	defer close(s.doneCh)
	<-s.waitCh
	s.fn()
}
func newStep(fn func()) *step {
	return &step{
		fn:     fn,
		waitCh: make(chan struct{}),
		doneCh: make(chan struct{}),
	}
}

func TestGracefulTerminationWithKeepListeningDuringGracefulTerminationDisabled(t *testing.T) {
	s := newGenericAPIServer(t, false)

	// record the termination events in the order they are signaled
	var signalOrderLock sync.Mutex
	signalOrderGot := make([]string, 0)
	recordOrderFn := func(before bool, name string, e lifecycleSignal) {
		if !before {
			return
		}
		signalOrderLock.Lock()
		defer signalOrderLock.Unlock()
		signalOrderGot = append(signalOrderGot, name)
	}

	// handler for a request that we want to keep in flight through to the end
	inFlightRequestBlockedCh, inFlightStartedCh := make(chan result), make(chan struct{})
	inFlightRequest := http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		close(inFlightStartedCh)
		// this request handler blocks until we deliberately unblock it.
		<-inFlightRequestBlockedCh
		w.WriteHeader(http.StatusOK)
	})
	s.Handler.NonGoRestfulMux.Handle("/in-flight-request-as-designed", inFlightRequest)

	connReusingClient := newClient(false)
	doer := setupDoer(t, s.SecureServingInfo)

	var delayedStopVerificationStepExecuted bool
	delayedStopVerificationStep := newStep(func() {
		delayedStopVerificationStepExecuted = true
		t.Log("Before ShutdownDelayDuration elapses new request(s) should be served")
		resultGot := doer.Do(connReusingClient, shouldReuseConnection(t), "/echo?message=request-on-an-existing-connection-should-succeed", time.Second)
		assertResponse(t, resultGot, http.StatusOK)
		resultGot = doer.Do(newClient(true), shouldUseNewConnection(t), "/echo?message=request-on-a-new-tcp-connection-should-succeed", time.Second)
		assertResponse(t, resultGot, http.StatusOK)
	})
	steps := func(before bool, name string, e lifecycleSignal) {
		// Before AfterShutdownDelayDuration event is signaled, the test
		// will send request(s) to assert on expected behavior.
		if name == "AfterShutdownDelayDuration" && before {
			// it unblocks the verification step and waits for it to complete
			<-delayedStopVerificationStep.done()
		}
	}

	// wrap the termination signals of the GenericAPIServer so the test can inject its own callback
	wrapLifecycleSignals(t, &s.lifecycleSignals, func(before bool, name string, e lifecycleSignal) {
		recordOrderFn(before, name, e)
		steps(before, name, e)
	})

	// start the API server
	stopCh, runCompletedCh := make(chan struct{}), make(chan struct{})
	go func() {
		defer close(runCompletedCh)
		s.PrepareRun().Run(stopCh)
	}()
	waitForAPIServerStarted(t, doer)

	// step 1: fire a request that we want to keep in-flight through to the end
	inFlightResultCh := make(chan result)
	go func() {
		resultGot := doer.Do(connReusingClient, func(httptrace.GotConnInfo) {}, "/in-flight-request-as-designed", 0)
		inFlightResultCh <- resultGot
	}()
	select {
	case <-inFlightStartedCh:
	case <-time.After(5 * time.Second):
		t.Fatalf("Waited for 5s for the in-flight request to reach the server")
	}

	// step 2: signal termination event: initiate a shutdown
	close(stopCh)

	// step 3: before ShutdownDelayDuration elapses new request(s) should be served successfully.
	delayedStopVerificationStep.execute()
	if !delayedStopVerificationStepExecuted {
		t.Fatal("Expected the AfterShutdownDelayDuration verification step to execute")
	}

	// step 4: wait for the HTTP Server listener to have stopped
	httpServerStoppedListeningCh := s.lifecycleSignals.HTTPServerStoppedListening
	select {
	case <-httpServerStoppedListeningCh.Signaled():
	case <-time.After(5 * time.Second):
		t.Fatal("Expected the server to signal HTTPServerStoppedListening event")
	}

	// step 5: the server has stopped listening but we still have a request
	// in flight, let it unblock and we expect the request to succeed.
	close(inFlightRequestBlockedCh)
	var inFlightResultGot result
	select {
	case inFlightResultGot = <-inFlightResultCh:
	case <-time.After(5 * time.Second):
		t.Fatal("Expected the server to send a response")
	}
	assertResponse(t, inFlightResultGot, http.StatusOK)

	t.Log("Waiting for the apiserver Run method to return")
	select {
	case <-runCompletedCh:
	case <-time.After(5 * time.Second):
		t.Fatal("Expected the apiserver Run method to return")
	}

	lifecycleSignalOrderExpected := []string{
		string("ShutdownInitiated"),
		string("AfterShutdownDelayDuration"),
		string("HTTPServerStoppedListening"),
		string("InFlightRequestsDrained"),
	}
	func() {
		signalOrderLock.Lock()
		defer signalOrderLock.Unlock()
		if !reflect.DeepEqual(lifecycleSignalOrderExpected, signalOrderGot) {
			t.Errorf("Expected order of termination event signal to match, diff: %s", cmp.Diff(lifecycleSignalOrderExpected, signalOrderGot))
		}
	}()
}

func TestGracefulTerminationWithKeepListeningDuringGracefulTerminationEnabled(t *testing.T) {
	s := newGenericAPIServer(t, true)

	// record the termination events in the order they are signaled
	var signalOrderLock sync.Mutex
	signalOrderGot := make([]string, 0)
	recordOrderFn := func(before bool, name string, e lifecycleSignal) {
		if !before {
			return
		}
		signalOrderLock.Lock()
		defer signalOrderLock.Unlock()
		signalOrderGot = append(signalOrderGot, name)
	}

	// handler for a request that we want to keep in flight through to the end
	inFlightRequestBlockedCh, inFlightStartedCh := make(chan result), make(chan struct{})
	inFlightRequest := http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		close(inFlightStartedCh)
		// this request handler blocks until we deliberately unblock it.
		<-inFlightRequestBlockedCh
		w.WriteHeader(http.StatusOK)
	})
	s.Handler.NonGoRestfulMux.Handle("/in-flight-request-as-designed", inFlightRequest)

	connReusingClient := newClient(false)
	doer := setupDoer(t, s.SecureServingInfo)

	var delayedStopVerificationStepExecuted bool
	delayedStopVerificationStep := newStep(func() {
		delayedStopVerificationStepExecuted = true
		t.Log("Before ShutdownDelayDuration elapses new request(s) should be served")
		resultGot := doer.Do(connReusingClient, shouldReuseConnection(t), "/echo?message=request-on-an-existing-connection-should-succeed", time.Second)
		assertResponse(t, resultGot, http.StatusOK)
		resultGot = doer.Do(newClient(true), shouldUseNewConnection(t), "/echo?message=request-on-a-new-tcp-connection-should-succeed", time.Second)
		assertResponse(t, resultGot, http.StatusOK)
	})
	steps := func(before bool, name string, e lifecycleSignal) {
		// Before AfterShutdownDelayDuration event is signaled, the test
		// will send request(s) to assert on expected behavior.
		if name == "AfterShutdownDelayDuration" && before {
			// it unblocks the verification step and waits for it to complete
			<-delayedStopVerificationStep.done()
		}
	}

	// wrap the termination signals of the GenericAPIServer so the test can inject its own callback
	wrapLifecycleSignals(t, &s.lifecycleSignals, func(before bool, name string, e lifecycleSignal) {
		recordOrderFn(before, name, e)
		steps(before, name, e)
	})

	// start the API server
	stopCh, runCompletedCh := make(chan struct{}), make(chan struct{})
	go func() {
		defer close(runCompletedCh)
		s.PrepareRun().Run(stopCh)
	}()
	waitForAPIServerStarted(t, doer)

	// step 1: fire a request that we want to keep in-flight through to the end
	inFlightResultCh := make(chan result)
	go func() {
		resultGot := doer.Do(connReusingClient, func(httptrace.GotConnInfo) {}, "/in-flight-request-as-designed", 0)
		inFlightResultCh <- resultGot
	}()
	select {
	case <-inFlightStartedCh:
	case <-time.After(5 * time.Second):
		t.Fatalf("Waited for 5s for the in-flight request to reach the server")
	}

	//step 1: /readyz should return OK
	resultGot := doer.Do(connReusingClient, func(httptrace.GotConnInfo) {}, "/readyz", time.Second)
	assertResponse(t, resultGot, http.StatusOK)

	// step 2: signal termination event: initiate a shutdown
	close(stopCh)

	// step 3: /readyz must return an error, but we need to give it some time
	err := wait.PollImmediate(100*time.Millisecond, 5*time.Second, func() (done bool, err error) {
		resultGot := doer.Do(connReusingClient, func(httptrace.GotConnInfo) {}, "/readyz", time.Second)
		// wait until we have a non 200 response
		if resultGot.response != nil && resultGot.response.StatusCode == http.StatusOK {
			return false, nil
		}

		assertResponse(t, resultGot, http.StatusInternalServerError)
		return true, nil
	})
	if err != nil {
		t.Errorf("Expected /readyz to return 500 status code, but got: %v", err)
	}

	// step 4: before ShutdownDelayDuration elapses new request(s) should be served successfully.
	delayedStopVerificationStep.execute()
	if !delayedStopVerificationStepExecuted {
		t.Fatal("Expected the AfterShutdownDelayDuration verification step to execute")
	}

	// step 5: ShutdownDelayDuration has elapsed, all incoming requests should receive 429
	t.Log("Verify that new incoming request(s) get 429")
	resultGot = doer.Do(newClient(true), shouldUseNewConnection(t), "/echo?message=request-on-a-new-tcp-connection-should-fail-with-429", time.Second)
	requestMustFailWithRetryHeader(t, resultGot, http.StatusTooManyRequests)
	resultGot = doer.Do(connReusingClient, shouldReuseConnection(t), "/echo?message=request-on-an-existing-connection-should-fail-with-429", time.Second)
	requestMustFailWithRetryHeader(t, resultGot, http.StatusTooManyRequests)

	// step 6: we still have a request in flight, let it unblock and we expect the request to succeed.
	close(inFlightRequestBlockedCh)
	var inFlightResultGot result
	select {
	case inFlightResultGot = <-inFlightResultCh:
	case <-time.After(5 * time.Second):
		t.Fatal("Expected the server to send a response")
	}
	assertResponse(t, inFlightResultGot, http.StatusOK)

	// step 7: wait for the HTTP Server listener to have stopped
	httpServerStoppedListeningCh := s.lifecycleSignals.HTTPServerStoppedListening
	select {
	case <-httpServerStoppedListeningCh.Signaled():
	case <-time.After(5 * time.Second):
		t.Fatal("Expected the server to signal HTTPServerStoppedListening event")
	}

	t.Log("Waiting for the apiserver Run method to return")
	select {
	case <-runCompletedCh:
	case <-time.After(5 * time.Second):
		t.Fatal("Expected the apiserver Run method to return")
	}

	lifecycleSignalOrderExpected := []string{
		string("ShutdownInitiated"),
		string("AfterShutdownDelayDuration"),
		string("InFlightRequestsDrained"),
		string("HTTPServerStoppedListening"),
	}
	func() {
		signalOrderLock.Lock()
		defer signalOrderLock.Unlock()
		if !reflect.DeepEqual(lifecycleSignalOrderExpected, signalOrderGot) {
			t.Errorf("Expected order of termination event signal to match, diff: %s", cmp.Diff(lifecycleSignalOrderExpected, signalOrderGot))
		}
	}()
}

func TestMuxAndDiscoveryComplete(t *testing.T) {
	// setup
	testSignal1 := make(chan struct{})
	testSignal2 := make(chan struct{})
	s := newGenericAPIServer(t, true)
	s.muxAndDiscoveryCompleteSignals["TestSignal1"] = testSignal1
	s.muxAndDiscoveryCompleteSignals["TestSignal2"] = testSignal2
	doer := setupDoer(t, s.SecureServingInfo)
	isChanClosed := func(ch <-chan struct{}, delay time.Duration) bool {
		time.Sleep(delay)
		select {
		case <-ch:
			return true
		default:
			return false
		}
	}

	// start the API server
	stopCh, runCompletedCh := make(chan struct{}), make(chan struct{})
	go func() {
		defer close(runCompletedCh)
		s.PrepareRun().Run(stopCh)
	}()
	waitForAPIServerStarted(t, doer)

	// act
	if isChanClosed(s.lifecycleSignals.MuxAndDiscoveryComplete.Signaled(), 1*time.Second) {
		t.Fatalf("%s is closed whereas the TestSignal is still open", s.lifecycleSignals.MuxAndDiscoveryComplete.Name())
	}

	close(testSignal1)
	if isChanClosed(s.lifecycleSignals.MuxAndDiscoveryComplete.Signaled(), 1*time.Second) {
		t.Fatalf("%s is closed whereas the TestSignal2 is still open", s.lifecycleSignals.MuxAndDiscoveryComplete.Name())
	}

	close(testSignal2)
	if !isChanClosed(s.lifecycleSignals.MuxAndDiscoveryComplete.Signaled(), 1*time.Second) {
		t.Fatalf("%s wasn't closed", s.lifecycleSignals.MuxAndDiscoveryComplete.Name())
	}
}
func shouldReuseConnection(t *testing.T) func(httptrace.GotConnInfo) {
	return func(ci httptrace.GotConnInfo) {
		if !ci.Reused {
			t.Errorf("Expected the request to use an existing TCP connection, but got: %+v", ci)
		}
	}
}

func shouldUseNewConnection(t *testing.T) func(httptrace.GotConnInfo) {
	return func(ci httptrace.GotConnInfo) {
		if ci.Reused {
			t.Errorf("Expected the request to use a new TCP connection, but got: %+v", ci)
		}
	}
}

func assertResponse(t *testing.T, resultGot result, statusCodeExpected int) {
	if resultGot.err != nil {
		t.Errorf("Expected no error, but got: %v", resultGot.err)
		return
	}
	if resultGot.response.StatusCode != statusCodeExpected {
		t.Errorf("Expected Status Code: %d, but got: %d", statusCodeExpected, resultGot.response.StatusCode)
	}
}

func requestMustFailWithRetryHeader(t *testing.T, resultGot result, statusCodedExpected int) {
	if resultGot.err != nil {
		t.Errorf("Expected no error, but got: %v", resultGot.err)
		return
	}
	if statusCodedExpected != resultGot.response.StatusCode {
		t.Errorf("Expected Status Code: %d, but got: %d", statusCodedExpected, resultGot.response.StatusCode)
	}
	retryAfterGot := resultGot.response.Header.Get("Retry-After")
	if retryAfterGot != "5" {
		t.Errorf("Expected Retry-After Response Header, but got: %v", resultGot.response)
	}
}

func waitForAPIServerStarted(t *testing.T, doer doer) {
	client := newClient(true)
	i := 1
	err := wait.PollImmediate(100*time.Millisecond, 5*time.Second, func() (done bool, err error) {
		result := doer.Do(client, func(httptrace.GotConnInfo) {}, fmt.Sprintf("/echo?message=attempt-%d", i), 100*time.Millisecond)
		i++

		if result.err != nil {
			t.Logf("Still waiting for the server to start - err: %v", err)
			return false, nil
		}
		if result.response.StatusCode != http.StatusOK {
			t.Logf("Still waiting for the server to start - expecting: %d, but got: %v", http.StatusOK, result.response)
			return false, nil
		}

		t.Log("The API server has started")
		return true, nil
	})

	if err != nil {
		t.Fatalf("The server has failed to start - err: %v", err)
	}
}

func setupDoer(t *testing.T, info *SecureServingInfo) doer {
	_, port, err := info.HostPort()
	if err != nil {
		t.Fatalf("Expected host, port from SecureServingInfo, but got: %v", err)
	}

	return func(client *http.Client, callback func(httptrace.GotConnInfo), path string, timeout time.Duration) result {
		url := fmt.Sprintf("https://%s:%d%s", "127.0.0.1", port, path)
		t.Logf("Sending request - timeout: %s, url: %s", timeout, url)

		req, err := http.NewRequest("GET", url, nil)
		if err != nil {
			return result{response: nil, err: err}
		}

		// setup request timeout
		var ctx context.Context
		if timeout > 0 {
			var cancel context.CancelFunc
			ctx, cancel = context.WithTimeout(req.Context(), timeout)
			defer cancel()

			req = req.WithContext(ctx)
		}

		// setup trace
		trace := &httptrace.ClientTrace{
			GotConn: func(connInfo httptrace.GotConnInfo) {
				callback(connInfo)
			},
		}
		req = req.WithContext(httptrace.WithClientTrace(req.Context(), trace))

		response, err := client.Do(req)
		// in this test, we don't depend on the body of the response, so we can
		// close the Body here to ensure the underlying transport can be reused
		if response != nil {
			ioutil.ReadAll(response.Body)
			response.Body.Close()
		}
		return result{
			err:      err,
			response: response,
		}
	}
}

func newClient(useNewConnection bool) *http.Client {
	clientCACertPool := x509.NewCertPool()
	clientCACertPool.AppendCertsFromPEM(backendCrt)
	tlsConfig := &tls.Config{
		RootCAs:    clientCACertPool,
		NextProtos: []string{http2.NextProtoTLS},
	}

	tr := &http.Transport{
		TLSClientConfig:   tlsConfig,
		DisableKeepAlives: useNewConnection,
	}
	if err := http2.ConfigureTransport(tr); err != nil {
		log.Fatalf("Failed to configure HTTP2 transport: %v", err)
	}
	return &http.Client{
		Timeout:   0,
		Transport: tr,
	}
}

func newGenericAPIServer(t *testing.T, keepListening bool) *GenericAPIServer {
	config, _ := setUp(t)
	config.ShutdownDelayDuration = 100 * time.Millisecond
	config.ShutdownSendRetryAfter = keepListening
	config.BuildHandlerChainFunc = func(apiHandler http.Handler, c *Config) http.Handler {
		handler := genericfilters.WithWaitGroup(apiHandler, c.LongRunningFunc, c.HandlerChainWaitGroup)
		if c.ShutdownSendRetryAfter {
			handler = genericfilters.WithRetryAfter(handler, c.lifecycleSignals.AfterShutdownDelayDuration.Signaled())
		}
		handler = genericapifilters.WithRequestInfo(handler, c.RequestInfoResolver)
		return handler
	}

	s, err := config.Complete(nil).New("test", NewEmptyDelegate())
	if err != nil {
		t.Fatalf("Error in bringing up the server: %v", err)
	}

	ln, err := net.Listen("tcp", "0.0.0.0:0")
	if err != nil {
		t.Fatalf("failed to listen on %v: %v", "0.0.0.0:0", err)
	}
	s.SecureServingInfo = &SecureServingInfo{}
	s.SecureServingInfo.Listener = &wrappedListener{ln, t}

	cert, err := dynamiccertificates.NewStaticCertKeyContent("serving-cert", backendCrt, backendKey)
	if err != nil {
		t.Fatalf("failed to load cert - %v", err)
	}
	s.SecureServingInfo.Cert = cert

	// we use this handler to send a test request to the server.
	s.Handler.NonGoRestfulMux.Handle("/echo", http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		t.Logf("[server] received a request, proto: %s, url: %s", req.Proto, req.RequestURI)

		w.Header().Add("echo", req.URL.Query().Get("message"))
		w.WriteHeader(http.StatusOK)
	}))

	return s
}

type wrappedListener struct {
	net.Listener
	t *testing.T
}

func (ln wrappedListener) Accept() (net.Conn, error) {
	c, err := ln.Listener.Accept()

	if tc, ok := c.(*net.TCPConn); ok {
		ln.t.Logf("[server] seen new connection: %#v", tc)
	}
	return c, err
}
