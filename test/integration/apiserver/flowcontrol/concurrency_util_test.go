/*
Copyright 2019 The Kubernetes Authors.

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

package flowcontrol

import (
	"context"
	"crypto/tls"
	"crypto/x509"
	"encoding/json"
	"fmt"
	"io"
	"io/ioutil"
	"math"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/prometheus/common/expfmt"
	"github.com/prometheus/common/model"

	authorizationv1 "k8s.io/api/authorization/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apiserver/pkg/admission/plugin/webhook/testcerts"
	"k8s.io/apiserver/pkg/authorization/authorizer"
	genericfeatures "k8s.io/apiserver/pkg/features"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	webhookutil "k8s.io/apiserver/pkg/util/webhook"
	"k8s.io/apiserver/plugin/pkg/authorizer/webhook"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	v1 "k8s.io/client-go/tools/clientcmd/api/v1"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/kubernetes/cmd/kube-apiserver/app/options"
	"k8s.io/kubernetes/pkg/controlplane"
	"k8s.io/kubernetes/test/integration/framework"
)

const (
	requestConcurrencyLimitMetricsName = "apiserver_flowcontrol_request_concurrency_limit"
	requestExecutionSecondsSumName     = "apiserver_flowcontrol_request_execution_seconds_sum"
	requestExecutionSecondsCountName   = "apiserver_flowcontrol_request_execution_seconds_count"
	priorityLevelSeatUtilSumName       = "apiserver_flowcontrol_priority_level_request_count_samples_sum"
	priorityLevelSeatUtilCountName     = "apiserver_flowcontrol_priority_level_request_count_samples_count"
	fakeworkDuration                   = 200 * time.Millisecond
	testWarmUpTime                     = 2 * time.Second
	testTime                           = 10 * time.Second
)

func setupWithAuthorizer(t testing.TB, maxReadonlyRequestsInFlight, maxMutatingRequestsInFlight int, authz authorizer.Authorizer) (*rest.Config, framework.TearDownFunc) {
	_, kubeConfig, tearDownFn := framework.StartTestServer(t, framework.TestServerSetup{
		ModifyServerRunOptions: func(opts *options.ServerRunOptions) {
			// Ensure all clients are allowed to send requests.
			opts.Authorization.Modes = []string{"AlwaysAllow"}
			opts.GenericServerRunOptions.MaxRequestsInFlight = maxReadonlyRequestsInFlight
			opts.GenericServerRunOptions.MaxMutatingRequestsInFlight = maxMutatingRequestsInFlight
		},
		ModifyServerConfig: func(config *controlplane.Config) {
			config.GenericConfig.Authorization.Authorizer = authz
		},
	})
	return kubeConfig, tearDownFn
}

func TestConcurrencyIsolation(t *testing.T) {
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, genericfeatures.APIPriorityAndFairness, true)()
	// NOTE: disabling the feature should fail the test
	// start webhook server
	serv := &mockV1Service{allow: true, statusCode: 200}
	s, err := NewV1TestServer(serv, testcerts.ServerCert, testcerts.ServerKey, testcerts.CACert)
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()

	authorizer, err := newV1Authorizer(s.URL, testcerts.ClientCert, testcerts.ClientKey, testcerts.CACert, 0)
	if err != nil {
		t.Fatal(err)
	}

	kubeConfig, closeFn := setupWithAuthorizer(t, 10, 10, authorizer)
	defer closeFn()

	loopbackClient := clientset.NewForConfigOrDie(kubeConfig)
	noxu1Client := getClientFor(kubeConfig, "noxu1")
	noxu2Client := getClientFor(kubeConfig, "noxu2")

	queueLength := 50
	concurrencyShares := 100

	priorityLevelNoxu1, _, err := createPriorityLevelAndBindingFlowSchemaForUser(
		loopbackClient, "noxu1", concurrencyShares, queueLength)
	if err != nil {
		t.Error(err)
	}
	priorityLevelNoxu2, _, err := createPriorityLevelAndBindingFlowSchemaForUser(
		loopbackClient, "noxu2", concurrencyShares, queueLength)
	if err != nil {
		t.Error(err)
	}
	availableSeats, err := getAvailableSeatsOfPriorityLevel(loopbackClient)
	if err != nil {
		t.Error(err)
	}

	t.Logf("noxu1 priority level concurrency limit: %v", availableSeats[priorityLevelNoxu1.Name])
	t.Logf("noxu2 priority level concurrency limit: %v", availableSeats[priorityLevelNoxu2.Name])
	if (availableSeats[priorityLevelNoxu1.Name] <= 4) || (availableSeats[priorityLevelNoxu2.Name] <= 4) {
		t.Errorf("The number of available seats for test client priority levels are too small: (%v, %v). Expecting a number > 4",
			availableSeats[priorityLevelNoxu1.Name], availableSeats[priorityLevelNoxu2.Name])
	}

	stopCh := make(chan struct{})
	wg := sync.WaitGroup{}

	// "elephant"
	noxu1NumGoroutines := 5 + queueLength
	var noxu1ClientRequestLatencySum float64
	var noxu1ClientRequestLatencySumSq float64
	var noxu1ClientRequestLatencyCount int32
	var noxu1Mutex sync.Mutex
	streamRequestsWithIndex(noxu1NumGoroutines, func(idx int) {
		start := time.Now()
		_, err := noxu1Client.CoreV1().Namespaces().Get(context.Background(), "default", metav1.GetOptions{})
		duration := time.Since(start).Seconds()
		noxu1Mutex.Lock()
		noxu1ClientRequestLatencyCount += 1
		noxu1ClientRequestLatencySum += duration
		noxu1ClientRequestLatencySumSq += duration * duration
		noxu1Mutex.Unlock()
		if err != nil {
			t.Error(err)
		}
	}, &wg, stopCh)
	// "mouse"
	noxu2NumGoroutines := 3
	var noxu2ClientRequestLatencySum float64
	var noxu2ClientRequestLatencySumSq float64
	var noxu2ClientRequestLatencyCount int32
	var noxu2Mutex sync.Mutex
	streamRequestsWithIndex(noxu2NumGoroutines, func(idx int) {
		start := time.Now()
		_, err := noxu2Client.CoreV1().Namespaces().Get(context.Background(), "default", metav1.GetOptions{})
		duration := time.Since(start).Seconds()
		noxu2Mutex.Lock()
		noxu2ClientRequestLatencyCount += 1
		noxu2ClientRequestLatencySum += duration
		noxu2ClientRequestLatencySumSq += duration * duration
		noxu2Mutex.Unlock()
		if err != nil {
			t.Error(err)
		}
	}, &wg, stopCh)

	// Warm up
	time.Sleep(testWarmUpTime)

	// Reset counters
	noxu1Mutex.Lock()
	noxu1ClientRequestLatencyCount = 0
	noxu1ClientRequestLatencySum = 0
	noxu1ClientRequestLatencySumSq = 0
	noxu1Mutex.Unlock()
	noxu2Mutex.Lock()
	noxu2ClientRequestLatencyCount = 0
	noxu2ClientRequestLatencySum = 0
	noxu2ClientRequestLatencySumSq = 0
	noxu2Mutex.Unlock()
	earlierRequestExecutionSecondsSum, earlierRequestExecutionSecondsCount, earlierPLSeatUtilSum, earlierPLSeatUtilCount, err := getRequestExecutionMetrics(loopbackClient)
	if err != nil {
		t.Error(err)
	}
	time.Sleep(testTime) // after warming up, the test enters a steady state
	laterRequestExecutionSecondsSum, laterRequestExecutionSecondsCount, laterPLSeatUtilSum, laterPLSeatUtilCount, err := getRequestExecutionMetrics(loopbackClient)
	if err != nil {
		t.Error(err)
	}
	if (earlierPLSeatUtilCount[priorityLevelNoxu1.Name] >= laterPLSeatUtilCount[priorityLevelNoxu1.Name]) || (earlierPLSeatUtilCount[priorityLevelNoxu2.Name] >= laterPLSeatUtilCount[priorityLevelNoxu2.Name]) {
		t.Errorf("PLSeatUtilCount check failed: noxu1 earlier count %v, later count %v; noxu2 earlier count %v, later count %v",
			earlierPLSeatUtilCount[priorityLevelNoxu1.Name], laterPLSeatUtilCount[priorityLevelNoxu1.Name], earlierPLSeatUtilCount[priorityLevelNoxu2.Name], laterPLSeatUtilCount[priorityLevelNoxu2.Name])
	}
	close(stopCh)

	noxu1RequestExecutionSecondsAvg := (laterRequestExecutionSecondsSum[priorityLevelNoxu1.Name] - earlierRequestExecutionSecondsSum[priorityLevelNoxu1.Name]) / float64(laterRequestExecutionSecondsCount[priorityLevelNoxu1.Name]-earlierRequestExecutionSecondsCount[priorityLevelNoxu1.Name])
	noxu2RequestExecutionSecondsAvg := (laterRequestExecutionSecondsSum[priorityLevelNoxu2.Name] - earlierRequestExecutionSecondsSum[priorityLevelNoxu2.Name]) / float64(laterRequestExecutionSecondsCount[priorityLevelNoxu2.Name]-earlierRequestExecutionSecondsCount[priorityLevelNoxu2.Name])
	noxu1PLSeatUtilAvg := (laterPLSeatUtilSum[priorityLevelNoxu1.Name] - earlierPLSeatUtilSum[priorityLevelNoxu1.Name]) / float64(laterPLSeatUtilCount[priorityLevelNoxu1.Name]-earlierPLSeatUtilCount[priorityLevelNoxu1.Name])
	noxu2PLSeatUtilAvg := (laterPLSeatUtilSum[priorityLevelNoxu2.Name] - earlierPLSeatUtilSum[priorityLevelNoxu2.Name]) / float64(laterPLSeatUtilCount[priorityLevelNoxu2.Name]-earlierPLSeatUtilCount[priorityLevelNoxu2.Name])
	t.Logf("\nnoxu1RequestExecutionSecondsAvg %v\nnoxu2RequestExecutionSecondsAvg %v", noxu1RequestExecutionSecondsAvg, noxu2RequestExecutionSecondsAvg)
	t.Logf("\nnoxu1PLSeatUtilAvg %v\nnoxu2PLSeatUtilAvg %v", noxu1PLSeatUtilAvg, noxu2PLSeatUtilAvg)

	wg.Wait() // wait till the client goroutines finish before computing the statistics
	noxu1ClientRequestLatencySecondsAvg, noxu1ClientRequestLatencySecondsSdev := computeClientRequestLatencyStats(noxu1ClientRequestLatencyCount, noxu1ClientRequestLatencySum, noxu1ClientRequestLatencySumSq)
	noxu2ClientRequestLatencySecondsAvg, noxu2ClientRequestLatencySecondsSdev := computeClientRequestLatencyStats(noxu2ClientRequestLatencyCount, noxu2ClientRequestLatencySum, noxu2ClientRequestLatencySumSq)
	t.Logf("\nnoxu1ClientRequestLatencyCount %v\nnoxu2ClientRequestLatencyCount %v", noxu1ClientRequestLatencyCount, noxu2ClientRequestLatencyCount)
	t.Logf("\nnoxu1ClientRequestLatencySecondsAvg %v\nnoxu2ClientRequestLatencySecondsAvg %v", noxu1ClientRequestLatencySecondsAvg, noxu2ClientRequestLatencySecondsAvg)
	t.Logf("\nnoxu1ClientRequestLatencySecondsSdev %v\nnoxu2ClientRequestLatencySecondsSdev %v", noxu1ClientRequestLatencySecondsSdev, noxu2ClientRequestLatencySecondsSdev)
	allDispatchedReqCounts, rejectedReqCounts, err := getRequestCountOfPriorityLevel(loopbackClient)
	if err != nil {
		t.Error(err)
	}
	t.Logf("\nnoxu1APFRequestCount %v\nnoxu2APFRequestCount %v", allDispatchedReqCounts[priorityLevelNoxu1.Name], allDispatchedReqCounts[priorityLevelNoxu2.Name])
	if rejectedReqCounts[priorityLevelNoxu1.Name] > 0 {
		t.Errorf(`%v requests from the "elephant" stream were rejected unexpectedly`, rejectedReqCounts[priorityLevelNoxu2.Name])
	}
	if rejectedReqCounts[priorityLevelNoxu2.Name] > 0 {
		t.Errorf(`%v requests from the "mouse" stream were rejected unexpectedly`, rejectedReqCounts[priorityLevelNoxu2.Name])
	}

	// Calculate server-side observed concurrency
	noxu1ObservedConcurrency := noxu1PLSeatUtilAvg * float64(availableSeats[priorityLevelNoxu1.Name])
	noxu2ObservedConcurrency := noxu2PLSeatUtilAvg * float64(availableSeats[priorityLevelNoxu2.Name])
	// Expected concurrency is derived from equal throughput assumption on both the client-side and the server-side
	// Expected concurrency computed can sometimes be larger than the number of available seats. We use the number of available seats as an upper bound
	noxu1ExpectedConcurrency := math.Min(float64(noxu1NumGoroutines)*noxu1RequestExecutionSecondsAvg/noxu1ClientRequestLatencySecondsAvg, float64(availableSeats[priorityLevelNoxu1.Name]))
	noxu2ExpectedConcurrency := math.Min(float64(noxu2NumGoroutines)*noxu2RequestExecutionSecondsAvg/noxu2ClientRequestLatencySecondsAvg, float64(availableSeats[priorityLevelNoxu2.Name]))
	t.Logf("Concurrency of noxu1:noxu2 - expected (%v:%v), observed (%v:%v)", noxu1ExpectedConcurrency, noxu2ExpectedConcurrency, noxu1ObservedConcurrency, noxu2ObservedConcurrency)
	// Calculate the tolerable error margin and perform the final check
	margin := 2 * math.Min(noxu1ClientRequestLatencySecondsSdev/noxu1ClientRequestLatencySecondsAvg, noxu2ClientRequestLatencySecondsSdev/noxu2ClientRequestLatencySecondsAvg)
	t.Logf("\nnoxu1Margin %v\nnoxu2Margin %v", noxu1ClientRequestLatencySecondsSdev/noxu1ClientRequestLatencySecondsAvg, noxu2ClientRequestLatencySecondsSdev/noxu2ClientRequestLatencySecondsAvg)
	t.Logf("Error margin is %v", margin)

	isConcurrencyExpected := func(name string, observed float64, expected float64) bool {
		t.Logf("%v relative error is %v", name, math.Abs(expected-observed)/expected)
		return math.Abs(expected-observed)/expected <= margin
	}
	if !isConcurrencyExpected(priorityLevelNoxu1.Name, noxu1ObservedConcurrency, noxu1ExpectedConcurrency) {
		t.Errorf("Concurrency observed by noxu1 is off. Expected: %v, observed: %v", noxu1ExpectedConcurrency, noxu1ObservedConcurrency)
	}
	if !isConcurrencyExpected(priorityLevelNoxu2.Name, noxu2ObservedConcurrency, noxu2ExpectedConcurrency) {
		t.Errorf("Concurrency observed by noxu2 is off. Expected: %v, observed: %v", noxu2ExpectedConcurrency, noxu2ObservedConcurrency)
	}

	// Check server-side APF measurements
	if math.Abs(1-noxu1PLSeatUtilAvg) > 0.05 {
		t.Errorf("noxu1PLSeatUtilAvg=%v is too far from expected=1.0", noxu1PLSeatUtilAvg)
	}
	if math.Abs(1-noxu2ObservedConcurrency/float64(noxu2NumGoroutines)) > 0.05 {
		t.Errorf("noxu2ObservedConcurrency=%v is too far from noxu2NumGoroutines=%v", noxu2ObservedConcurrency, noxu2NumGoroutines)
	}
}

func computeClientRequestLatencyStats(count int32, sum, sumsq float64) (float64, float64) {
	mean := sum / float64(count)
	ss := sumsq - mean*sum // reduced from ss := sumsq - 2*mean*sum + float64(count)*mean*mean
	return mean, math.Sqrt(ss / float64(count))
}

func getAvailableSeatsOfPriorityLevel(c clientset.Interface) (map[string]int, error) {
	resp, err := getMetrics(c)
	if err != nil {
		return nil, err
	}

	dec := expfmt.NewDecoder(strings.NewReader(string(resp)), expfmt.FmtText)
	decoder := expfmt.SampleDecoder{
		Dec:  dec,
		Opts: &expfmt.DecodeOptions{},
	}

	concurrency := make(map[string]int)
	for {
		var v model.Vector
		if err := decoder.Decode(&v); err != nil {
			if err == io.EOF {
				// Expected loop termination condition.
				return concurrency, nil
			}
			return nil, fmt.Errorf("failed decoding metrics: %v", err)
		}
		for _, metric := range v {
			switch name := string(metric.Metric[model.MetricNameLabel]); name {
			case requestConcurrencyLimitMetricsName:
				concurrency[string(metric.Metric[labelPriorityLevel])] = int(metric.Value)
			}
		}
	}
}

func getRequestExecutionMetrics(c clientset.Interface) (map[string]float64, map[string]int, map[string]float64, map[string]int, error) {

	resp, err := getMetrics(c)
	if err != nil {
		return nil, nil, nil, nil, err
	}

	dec := expfmt.NewDecoder(strings.NewReader(string(resp)), expfmt.FmtText)
	decoder := expfmt.SampleDecoder{
		Dec:  dec,
		Opts: &expfmt.DecodeOptions{},
	}

	RequestExecutionSecondsSum := make(map[string]float64)
	RequestExecutionSecondsCount := make(map[string]int)
	PLSeatUtilSum := make(map[string]float64)
	PLSeatUtilCount := make(map[string]int)

	for {
		var v model.Vector
		if err := decoder.Decode(&v); err != nil {
			if err == io.EOF {
				// Expected loop termination condition.
				return RequestExecutionSecondsSum, RequestExecutionSecondsCount,
					PLSeatUtilSum, PLSeatUtilCount, nil
			}
			return nil, nil, nil, nil, fmt.Errorf("failed decoding metrics: %v", err)
		}
		for _, metric := range v {
			switch name := string(metric.Metric[model.MetricNameLabel]); name {
			case requestExecutionSecondsSumName:
				RequestExecutionSecondsSum[string(metric.Metric[labelPriorityLevel])] = float64(metric.Value)
			case requestExecutionSecondsCountName:
				RequestExecutionSecondsCount[string(metric.Metric[labelPriorityLevel])] = int(metric.Value)
			case priorityLevelSeatUtilSumName:
				PLSeatUtilSum[string(metric.Metric[labelPriorityLevel])] = float64(metric.Value)
			case priorityLevelSeatUtilCountName:
				PLSeatUtilCount[string(metric.Metric[labelPriorityLevel])] = int(metric.Value)
			}
		}
	}
}

func streamRequestsWithIndex(parallel int, request func(idx int), wg *sync.WaitGroup, stopCh <-chan struct{}) {
	wg.Add(parallel)
	for i := 0; i < parallel; i++ {
		go func(idx int) {
			defer wg.Done()
			for {
				select {
				case <-stopCh:
					return
				default:
					request(idx)
				}
			}
		}(i)
	}
}

// Webhook authorizer code copied from staging/src/k8s.io/apiserver/plugin/pkg/authenticator/token/webhook/webhook_v1_test.go with minor changes
// V1Service mocks a remote service.
type V1Service interface {
	Review(*authorizationv1.SubjectAccessReview)
	HTTPStatusCode() int
}

// NewV1TestServer wraps a V1Service as an httptest.Server.
func NewV1TestServer(s V1Service, cert, key, caCert []byte) (*httptest.Server, error) {
	const webhookPath = "/testserver"
	var tlsConfig *tls.Config
	if cert != nil {
		cert, err := tls.X509KeyPair(cert, key)
		if err != nil {
			return nil, err
		}
		tlsConfig = &tls.Config{Certificates: []tls.Certificate{cert}}
	}

	if caCert != nil {
		rootCAs := x509.NewCertPool()
		rootCAs.AppendCertsFromPEM(caCert)
		if tlsConfig == nil {
			tlsConfig = &tls.Config{}
		}
		tlsConfig.ClientCAs = rootCAs
		tlsConfig.ClientAuth = tls.RequireAndVerifyClientCert
	}

	serveHTTP := func(w http.ResponseWriter, r *http.Request) {
		if r.Method != "POST" {
			http.Error(w, fmt.Sprintf("unexpected method: %v", r.Method), http.StatusMethodNotAllowed)
			return
		}
		if r.URL.Path != webhookPath {
			http.Error(w, fmt.Sprintf("unexpected path: %v", r.URL.Path), http.StatusNotFound)
			return
		}

		var review authorizationv1.SubjectAccessReview
		bodyData, _ := ioutil.ReadAll(r.Body)
		if err := json.Unmarshal(bodyData, &review); err != nil {
			http.Error(w, fmt.Sprintf("failed to decode body: %v", err), http.StatusBadRequest)
			return
		}

		// ensure we received the serialized review as expected
		if review.APIVersion != "authorization.k8s.io/v1" {
			http.Error(w, fmt.Sprintf("wrong api version: %s", string(bodyData)), http.StatusBadRequest)
			return
		}
		// once we have a successful request, always call the review to record that we were called
		s.Review(&review)
		if s.HTTPStatusCode() < 200 || s.HTTPStatusCode() >= 300 {
			http.Error(w, "HTTP Error", s.HTTPStatusCode())
			return
		}
		type status struct {
			Allowed         bool   `json:"allowed"`
			Reason          string `json:"reason"`
			EvaluationError string `json:"evaluationError"`
		}
		resp := struct {
			APIVersion string `json:"apiVersion"`
			Status     status `json:"status"`
		}{
			APIVersion: authorizationv1.SchemeGroupVersion.String(),
			Status:     status{review.Status.Allowed, review.Status.Reason, review.Status.EvaluationError},
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(resp)
	}

	server := httptest.NewUnstartedServer(http.HandlerFunc(serveHTTP))
	server.TLS = tlsConfig
	server.StartTLS()

	// Adjust the path to point to our custom path
	serverURL, _ := url.Parse(server.URL)
	serverURL.Path = webhookPath
	server.URL = serverURL.String()

	return server, nil
}

// A service that can be set to allow all or deny all authorization requests.
type mockV1Service struct {
	allow      bool
	statusCode int
	called     int
}

func (m *mockV1Service) Review(r *authorizationv1.SubjectAccessReview) {
	if r.Spec.User == "noxu1" || r.Spec.User == "noxu2" {
		time.Sleep(fakeworkDuration) // simulate fake work with sleep
	}
	m.called++
	r.Status.Allowed = m.allow
}
func (m *mockV1Service) HTTPStatusCode() int { return m.statusCode }

// newV1Authorizer creates a temporary kubeconfig file from the provided arguments and attempts to load
// a new WebhookAuthorizer from it.
func newV1Authorizer(callbackURL string, clientCert, clientKey, ca []byte, cacheTime time.Duration) (*webhook.WebhookAuthorizer, error) {
	tempfile, err := ioutil.TempFile("", "")
	if err != nil {
		return nil, err
	}
	p := tempfile.Name()
	defer os.Remove(p)
	config := v1.Config{
		Clusters: []v1.NamedCluster{
			{
				Cluster: v1.Cluster{Server: callbackURL, CertificateAuthorityData: ca},
			},
		},
		AuthInfos: []v1.NamedAuthInfo{
			{
				AuthInfo: v1.AuthInfo{ClientCertificateData: clientCert, ClientKeyData: clientKey},
			},
		},
	}
	if err := json.NewEncoder(tempfile).Encode(config); err != nil {
		return nil, err
	}
	clientConfig, err := webhookutil.LoadKubeconfig(p, nil)
	if err != nil {
		return nil, err
	}

	return webhook.New(clientConfig, "v1", cacheTime, cacheTime, testRetryBackoff)
}

var testRetryBackoff = wait.Backoff{
	Duration: 5 * time.Millisecond,
	Factor:   1.5,
	Jitter:   0.2,
	Steps:    5,
}
