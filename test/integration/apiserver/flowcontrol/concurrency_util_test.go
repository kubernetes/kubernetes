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
	priorityLevelSeatUtilSumName       = "apiserver_flowcontrol_priority_level_seat_utilization_sum"
	priorityLevelSeatUtilCountName     = "apiserver_flowcontrol_priority_level_seat_utilization_count"
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

type SumAndCount struct {
	Sum   float64
	Count int
}

type plMetrics struct {
	execSeconds    SumAndCount
	seatUtil       SumAndCount
	availableSeats int
}

// metricSnapshot maps from a priority level label to
// a plMetrics struct containing APF metrics of interest
type metricSnapshot map[string]plMetrics

// Client request latency measurement
type clientLatencyMeasurement struct {
	SumAndCount
	SumSq float64 // latency sum of squares
	Mu    sync.Mutex
}

func (clm *clientLatencyMeasurement) reset() {
	clm.Mu.Lock()
	defer clm.Mu.Unlock()
	clm.Sum = 0
	clm.Count = 0
	clm.SumSq = 0
}

func (clm *clientLatencyMeasurement) update(duration float64) {
	clm.Mu.Lock()
	defer clm.Mu.Unlock()
	clm.Count += 1
	clm.Sum += duration
	clm.SumSq += duration * duration
}

func (clm *clientLatencyMeasurement) getStats() clientLatencyStats {
	clm.Mu.Lock()
	defer clm.Mu.Unlock()
	mean := clm.Sum / float64(clm.Count)
	ss := clm.SumSq - mean*clm.Sum // reduced from ss := sumsq - 2*mean*sum + float64(count)*mean*mean
	// Set ss to 0 if negative value is resulted from floating point calculations
	if ss < 0 {
		ss = 0
	}
	stdDev := math.Sqrt(ss / float64(clm.Count))
	cv := stdDev / mean
	return clientLatencyStats{mean: mean, stdDev: stdDev, cv: cv}
}

type clientLatencyStats struct {
	mean   float64 // latency average
	stdDev float64 // latency population standard deviation
	cv     float64 // latency coefficient of variation
}

type plMetricAvg struct {
	reqExecution float64 // average request execution time
	seatUtil     float64 // average seat utilization
}

func intervalMetricAvg(snapshot0, snapshot1 metricSnapshot, plLabel string) plMetricAvg {
	plmT0 := snapshot0[plLabel]
	plmT1 := snapshot1[plLabel]
	return plMetricAvg{
		reqExecution: (plmT1.execSeconds.Sum - plmT0.execSeconds.Sum) / float64(plmT1.execSeconds.Count-plmT0.execSeconds.Count),
		seatUtil:     (plmT1.seatUtil.Sum - plmT0.seatUtil.Sum) / float64(plmT1.seatUtil.Count-plmT0.seatUtil.Count),
	}
}

// TestConcurrencyIsolation tests the concurrency isolation between priority levels.
// The test defines two priority levels for this purpose, and corresponding flow schemas.
// To one priority level, this test sends many more concurrent requests than the configuration
// allows to execute at once, while sending fewer than allowed to the other priority level.
// The primary check is that the low flow gets all the seats it wants, but is modulated by
// recognizing that there are uncontrolled overheads in the system.
//
// This test differs from TestPriorityLevelIsolation since TestPriorityLevelIsolation checks throughput instead
// of concurrency. In order to mitigate the effects of system noise, authorization webhook is used to artificially
// increase request execution time to make the system noise relatively insignificant.
// Secondarily, this test also checks the observed seat utilizations against values derived from expecting that
// the throughput observed by the client equals the execution throughput observed by the server.
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

	plNoxu1, _, err := createPriorityLevelAndBindingFlowSchemaForUser(
		loopbackClient, "noxu1", concurrencyShares, queueLength)
	if err != nil {
		t.Error(err)
	}
	plNoxu2, _, err := createPriorityLevelAndBindingFlowSchemaForUser(
		loopbackClient, "noxu2", concurrencyShares, queueLength)
	if err != nil {
		t.Error(err)
	}

	stopCh := make(chan struct{})
	wg := sync.WaitGroup{}

	// "elephant"
	noxu1NumGoroutines := 5 + queueLength
	var noxu1LatMeasure clientLatencyMeasurement
	wg.Add(noxu1NumGoroutines)
	streamRequests(noxu1NumGoroutines, func() {
		start := time.Now()
		_, err := noxu1Client.CoreV1().Namespaces().Get(context.Background(), "default", metav1.GetOptions{})
		duration := time.Since(start).Seconds()
		noxu1LatMeasure.update(duration)
		if err != nil {
			t.Error(err)
		}
	}, &wg, stopCh)
	// "mouse"
	noxu2NumGoroutines := 3
	var noxu2LatMeasure clientLatencyMeasurement
	wg.Add(noxu2NumGoroutines)
	streamRequests(noxu2NumGoroutines, func() {
		start := time.Now()
		_, err := noxu2Client.CoreV1().Namespaces().Get(context.Background(), "default", metav1.GetOptions{})
		duration := time.Since(start).Seconds()
		noxu2LatMeasure.update(duration)
		if err != nil {
			t.Error(err)
		}
	}, &wg, stopCh)

	// Warm up
	time.Sleep(testWarmUpTime)

	noxu1LatMeasure.reset()
	noxu2LatMeasure.reset()
	snapshot0, err := getRequestMetricsSnapshot(loopbackClient)
	if err != nil {
		t.Error(err)
	}
	time.Sleep(testTime) // after warming up, the test enters a steady state
	snapshot1, err := getRequestMetricsSnapshot(loopbackClient)
	if err != nil {
		t.Error(err)
	}
	close(stopCh)

	// Check the assumptions of the test
	noxu1T0 := snapshot0[plNoxu1.Name]
	noxu1T1 := snapshot1[plNoxu1.Name]
	noxu2T0 := snapshot0[plNoxu2.Name]
	noxu2T1 := snapshot1[plNoxu2.Name]
	if noxu1T0.seatUtil.Count >= noxu1T1.seatUtil.Count || noxu2T0.seatUtil.Count >= noxu2T1.seatUtil.Count {
		t.Errorf("SeatUtilCount check failed: noxu1 t0 count %d, t1 count %d; noxu2 t0 count %d, t1 count %d",
			noxu1T0.seatUtil.Count, noxu1T1.seatUtil.Count, noxu2T0.seatUtil.Count, noxu2T1.seatUtil.Count)
	}
	t.Logf("noxu1 priority level concurrency limit: %d", noxu1T0.availableSeats)
	t.Logf("noxu2 priority level concurrency limit: %d", noxu2T0.availableSeats)
	if (noxu1T0.availableSeats != noxu1T1.availableSeats) || (noxu2T0.availableSeats != noxu2T1.availableSeats) {
		t.Errorf("The number of available seats changed: noxu1 (%d, %d) noxu2 (%d, %d)",
			noxu1T0.availableSeats, noxu1T1.availableSeats, noxu2T0.availableSeats, noxu2T1.availableSeats)
	}
	if (noxu1T0.availableSeats <= 4) || (noxu2T0.availableSeats <= 4) {
		t.Errorf("The number of available seats for test client priority levels are too small: (%d, %d). Expecting a number > 4",
			noxu1T0.availableSeats, noxu2T0.availableSeats)
	}
	// No requests should be rejected under normal situations
	_, rejectedReqCounts, err := getRequestCountOfPriorityLevel(loopbackClient)
	if err != nil {
		t.Error(err)
	}
	if rejectedReqCounts[plNoxu1.Name] > 0 {
		t.Errorf(`%d requests from the "elephant" stream were rejected unexpectedly`, rejectedReqCounts[plNoxu1.Name])
	}
	if rejectedReqCounts[plNoxu2.Name] > 0 {
		t.Errorf(`%d requests from the "mouse" stream were rejected unexpectedly`, rejectedReqCounts[plNoxu2.Name])
	}

	// Calculate APF server side metric averages during the test interval
	noxu1Avg := intervalMetricAvg(snapshot0, snapshot1, plNoxu1.Name)
	noxu2Avg := intervalMetricAvg(snapshot0, snapshot1, plNoxu2.Name)
	t.Logf("\nnoxu1 avg request execution time %v\nnoxu2 avg request execution time %v", noxu1Avg.reqExecution, noxu2Avg.reqExecution)
	t.Logf("\nnoxu1 avg seat utilization %v\nnoxu2 avg seat utilization %v", noxu1Avg.seatUtil, noxu2Avg.seatUtil)

	// Wait till the client goroutines finish before computing the client side request latency statistics
	wg.Wait()
	noxu1LatStats := noxu1LatMeasure.getStats()
	noxu2LatStats := noxu2LatMeasure.getStats()
	t.Logf("noxu1 client request count %d duration mean %v stddev %v cv %v", noxu1LatMeasure.Count, noxu1LatStats.mean, noxu1LatStats.stdDev, noxu1LatStats.cv)
	t.Logf("noxu2 client request count %d duration mean %v stddev %v cv %v", noxu2LatMeasure.Count, noxu2LatStats.mean, noxu2LatStats.stdDev, noxu2LatStats.cv)

	// Calculate server-side observed concurrency
	noxu1ObservedConcurrency := noxu1Avg.seatUtil * float64(noxu1T0.availableSeats)
	noxu2ObservedConcurrency := noxu2Avg.seatUtil * float64(noxu2T0.availableSeats)
	// Expected concurrency is derived from equal throughput assumption on both the client-side and the server-side
	noxu1ExpectedConcurrency := float64(noxu1NumGoroutines) * noxu1Avg.reqExecution / noxu1LatStats.mean
	noxu2ExpectedConcurrency := float64(noxu2NumGoroutines) * noxu2Avg.reqExecution / noxu2LatStats.mean
	t.Logf("Concurrency of noxu1:noxu2 - expected (%v:%v), observed (%v:%v)", noxu1ExpectedConcurrency, noxu2ExpectedConcurrency, noxu1ObservedConcurrency, noxu2ObservedConcurrency)

	// There are uncontrolled overheads that introduce noise into the system. The coefficient of variation (CV), that is,
	// standard deviation divided by mean, for a class of traffic is a characterization of all the noise that applied to
	// that class. We found that noxu1 generally had a much bigger CV than noxu2. This makes sense, because noxu1 probes
	// more behavior --- the waiting in queues. So we take the minimum of the two as an indicator of the relative amount
	// of noise that comes from all the other behavior. Currently, we use 2 times the experienced coefficient of variation
	// as the margin of error.
	margin := 2 * math.Min(noxu1LatStats.cv, noxu2LatStats.cv)
	t.Logf("Error margin is %v", margin)

	isConcurrencyExpected := func(name string, observed float64, expected float64) bool {
		relativeErr := math.Abs(expected-observed) / expected
		t.Logf("%v relative error is %v", name, relativeErr)
		return relativeErr <= margin
	}
	if !isConcurrencyExpected(plNoxu1.Name, noxu1ObservedConcurrency, noxu1ExpectedConcurrency) {
		t.Errorf("Concurrency observed by noxu1 is off. Expected: %v, observed: %v", noxu1ExpectedConcurrency, noxu1ObservedConcurrency)
	}
	if !isConcurrencyExpected(plNoxu2.Name, noxu2ObservedConcurrency, noxu2ExpectedConcurrency) {
		t.Errorf("Concurrency observed by noxu2 is off. Expected: %v, observed: %v", noxu2ExpectedConcurrency, noxu2ObservedConcurrency)
	}

	// Check the server-side APF seat utilization measurements
	if math.Abs(1-noxu1Avg.seatUtil) > 0.05 {
		t.Errorf("noxu1Avg.seatUtil=%v is too far from expected=1.0", noxu1Avg.seatUtil)
	}
	noxu2ExpectedSeatUtil := float64(noxu2NumGoroutines) / float64(noxu2T0.availableSeats)
	if math.Abs(noxu2ExpectedSeatUtil-noxu2Avg.seatUtil) > 0.05 {
		t.Errorf("noxu2Avg.seatUtil=%v is too far from expected=%v", noxu2Avg.seatUtil, noxu2ExpectedSeatUtil)
	}
}

func getRequestMetricsSnapshot(c clientset.Interface) (metricSnapshot, error) {

	resp, err := getMetrics(c)
	if err != nil {
		return nil, err
	}

	dec := expfmt.NewDecoder(strings.NewReader(string(resp)), expfmt.FmtText)
	decoder := expfmt.SampleDecoder{
		Dec:  dec,
		Opts: &expfmt.DecodeOptions{},
	}

	snapshot := metricSnapshot{}

	for {
		var v model.Vector
		if err := decoder.Decode(&v); err != nil {
			if err == io.EOF {
				// Expected loop termination condition.
				return snapshot, nil
			}
			return nil, fmt.Errorf("failed decoding metrics: %v", err)
		}
		for _, metric := range v {
			plLabel := string(metric.Metric[labelPriorityLevel])
			entry := plMetrics{}
			if v, ok := snapshot[plLabel]; ok {
				entry = v
			}
			switch name := string(metric.Metric[model.MetricNameLabel]); name {
			case requestExecutionSecondsSumName:
				entry.execSeconds.Sum = float64(metric.Value)
			case requestExecutionSecondsCountName:
				entry.execSeconds.Count = int(metric.Value)
			case priorityLevelSeatUtilSumName:
				entry.seatUtil.Sum = float64(metric.Value)
			case priorityLevelSeatUtilCountName:
				entry.seatUtil.Count = int(metric.Value)
			case requestConcurrencyLimitMetricsName:
				entry.availableSeats = int(metric.Value)
			}
			snapshot[plLabel] = entry
		}
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
}

func (m *mockV1Service) Review(r *authorizationv1.SubjectAccessReview) {
	if r.Spec.User == "noxu1" || r.Spec.User == "noxu2" {
		time.Sleep(fakeworkDuration) // simulate fake work with sleep
	}
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
