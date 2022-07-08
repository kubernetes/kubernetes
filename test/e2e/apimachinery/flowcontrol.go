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

package apimachinery

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"sync"
	"sync/atomic"
	"time"

	"github.com/onsi/ginkgo/v2"
	"github.com/prometheus/common/expfmt"
	"github.com/prometheus/common/model"

	flowcontrol "k8s.io/api/flowcontrol/v1beta2"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apiserver/pkg/util/apihelpers"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	clientsideflowcontrol "k8s.io/client-go/util/flowcontrol"
	"k8s.io/kubernetes/test/e2e/framework"
	admissionapi "k8s.io/pod-security-admission/api"
)

const (
	requestConcurrencyLimitMetricName = "apiserver_flowcontrol_request_concurrency_limit"
	priorityLevelLabelName            = "priority_level"
)

var (
	errPriorityLevelNotFound = errors.New("cannot find a metric sample with a matching priority level name label")
)

var _ = SIGDescribe("API priority and fairness", func() {
	f := framework.NewDefaultFramework("apf")
	f.NamespacePodSecurityEnforceLevel = admissionapi.LevelPrivileged

	ginkgo.It("should ensure that requests can be classified by adding FlowSchema and PriorityLevelConfiguration", func() {
		testingFlowSchemaName := "e2e-testing-flowschema"
		testingPriorityLevelName := "e2e-testing-prioritylevel"
		matchingUsername := "noxu"
		nonMatchingUsername := "foo"

		ginkgo.By("creating a testing PriorityLevelConfiguration object")
		createdPriorityLevel, cleanup := createPriorityLevel(f, testingPriorityLevelName, 1)
		defer cleanup()

		ginkgo.By("creating a testing FlowSchema object")
		createdFlowSchema, cleanup := createFlowSchema(f, testingFlowSchemaName, 1000, testingPriorityLevelName, []string{matchingUsername})
		defer cleanup()

		ginkgo.By("waiting for testing FlowSchema and PriorityLevelConfiguration to reach steady state")
		waitForSteadyState(f, testingFlowSchemaName, testingPriorityLevelName)

		var response *http.Response
		ginkgo.By("response headers should contain the UID of the appropriate FlowSchema and PriorityLevelConfiguration for a matching user")
		response = makeRequest(f, matchingUsername)
		if plUIDWant, plUIDGot := string(createdPriorityLevel.UID), getPriorityLevelUID(response); plUIDWant != plUIDGot {
			framework.Failf("expected PriorityLevelConfiguration UID in the response header: %s, but got: %s, response header: %#v", plUIDWant, plUIDGot, response.Header)
		}
		if fsUIDWant, fsUIDGot := string(createdFlowSchema.UID), getFlowSchemaUID(response); fsUIDWant != fsUIDGot {
			framework.Failf("expected FlowSchema UID in the response header: %s, but got: %s, response header: %#v", fsUIDWant, fsUIDGot, response.Header)
		}

		ginkgo.By("response headers should contain non-empty UID of FlowSchema and PriorityLevelConfiguration for a non-matching user")
		response = makeRequest(f, nonMatchingUsername)
		if plUIDGot := getPriorityLevelUID(response); plUIDGot == "" {
			framework.Failf("expected a non-empty PriorityLevelConfiguration UID in the response header, but got: %s, response header: %#v", plUIDGot, response.Header)
		}
		if fsUIDGot := getFlowSchemaUID(response); fsUIDGot == "" {
			framework.Failf("expected a non-empty FlowSchema UID in the response header but got: %s, response header: %#v", fsUIDGot, response.Header)
		}
	})

	// This test creates two flow schemas and a corresponding priority level for
	// each flow schema. One flow schema has a higher match precedence. With two
	// clients making requests at different rates, we test to make sure that the
	// higher QPS client cannot drown out the other one despite having higher
	// priority.
	ginkgo.It("should ensure that requests can't be drowned out (priority)", func() {
		// See https://github.com/kubernetes/kubernetes/issues/96710
		ginkgo.Skip("skipping test until flakiness is resolved")

		flowSchemaNamePrefix := "e2e-testing-flowschema-" + f.UniqueName
		priorityLevelNamePrefix := "e2e-testing-prioritylevel-" + f.UniqueName
		loadDuration := 10 * time.Second
		highQPSClientName := "highqps-" + f.UniqueName
		lowQPSClientName := "lowqps-" + f.UniqueName

		type client struct {
			username                    string
			qps                         float64
			priorityLevelName           string  //lint:ignore U1000 field is actually used
			concurrencyMultiplier       float64 //lint:ignore U1000 field is actually used
			concurrency                 int32
			flowSchemaName              string //lint:ignore U1000 field is actually used
			matchingPrecedence          int32  //lint:ignore U1000 field is actually used
			completedRequests           int32
			expectedCompletedPercentage float64 //lint:ignore U1000 field is actually used
		}
		clients := []client{
			// "highqps" refers to a client that creates requests at a much higher
			// QPS than its counter-part and well above its concurrency share limit.
			// In contrast, "lowqps" stays under its concurrency shares.
			// Additionally, the "highqps" client also has a higher matching
			// precedence for its flow schema.
			{username: highQPSClientName, qps: 90, concurrencyMultiplier: 2.0, matchingPrecedence: 999, expectedCompletedPercentage: 0.90},
			{username: lowQPSClientName, qps: 4, concurrencyMultiplier: 0.5, matchingPrecedence: 1000, expectedCompletedPercentage: 0.90},
		}

		ginkgo.By("creating test priority levels and flow schemas")
		for i := range clients {
			clients[i].priorityLevelName = fmt.Sprintf("%s-%s", priorityLevelNamePrefix, clients[i].username)
			framework.Logf("creating PriorityLevel %q", clients[i].priorityLevelName)
			_, cleanup := createPriorityLevel(f, clients[i].priorityLevelName, 1)
			defer cleanup()

			clients[i].flowSchemaName = fmt.Sprintf("%s-%s", flowSchemaNamePrefix, clients[i].username)
			framework.Logf("creating FlowSchema %q", clients[i].flowSchemaName)
			_, cleanup = createFlowSchema(f, clients[i].flowSchemaName, clients[i].matchingPrecedence, clients[i].priorityLevelName, []string{clients[i].username})
			defer cleanup()

			ginkgo.By("waiting for testing FlowSchema and PriorityLevelConfiguration to reach steady state")
			waitForSteadyState(f, clients[i].flowSchemaName, clients[i].priorityLevelName)
		}

		ginkgo.By("getting request concurrency from metrics")
		for i := range clients {
			realConcurrency, err := getPriorityLevelConcurrency(f.ClientSet, clients[i].priorityLevelName)
			framework.ExpectNoError(err)
			clients[i].concurrency = int32(float64(realConcurrency) * clients[i].concurrencyMultiplier)
			if clients[i].concurrency < 1 {
				clients[i].concurrency = 1
			}
			framework.Logf("request concurrency for %q will be %d (that is %d times client multiplier)", clients[i].username, clients[i].concurrency, realConcurrency)
		}

		ginkgo.By(fmt.Sprintf("starting uniform QPS load for %s", loadDuration.String()))
		var wg sync.WaitGroup
		for i := range clients {
			wg.Add(1)
			go func(c *client) {
				defer wg.Done()
				framework.Logf("starting uniform QPS load for %q: concurrency=%d, qps=%.1f", c.username, c.concurrency, c.qps)
				c.completedRequests = uniformQPSLoadConcurrent(f, c.username, c.concurrency, c.qps, loadDuration)
			}(&clients[i])
		}
		wg.Wait()

		ginkgo.By("checking completed requests with expected values")
		for _, client := range clients {
			// Each client should have 95% of its ideal number of completed requests.
			maxCompletedRequests := float64(client.concurrency) * client.qps * loadDuration.Seconds()
			fractionCompleted := float64(client.completedRequests) / maxCompletedRequests
			framework.Logf("client %q completed %d/%d requests (%.1f%%)", client.username, client.completedRequests, int32(maxCompletedRequests), 100*fractionCompleted)
			if fractionCompleted < client.expectedCompletedPercentage {
				framework.Failf("client %q: got %.1f%% completed requests, want at least %.1f%%", client.username, 100*fractionCompleted, 100*client.expectedCompletedPercentage)
			}
		}
	})

	// This test has two clients (different usernames) making requests at
	// different rates. Both clients' requests get mapped to the same flow schema
	// and priority level. We expect APF's "ByUser" flow distinguisher to isolate
	// the two clients and not allow one client to drown out the other despite
	// having a higher QPS.
	ginkgo.It("should ensure that requests can't be drowned out (fairness)", func() {
		// See https://github.com/kubernetes/kubernetes/issues/96710
		ginkgo.Skip("skipping test until flakiness is resolved")

		priorityLevelName := "e2e-testing-prioritylevel-" + f.UniqueName
		flowSchemaName := "e2e-testing-flowschema-" + f.UniqueName
		loadDuration := 10 * time.Second

		framework.Logf("creating PriorityLevel %q", priorityLevelName)
		_, cleanup := createPriorityLevel(f, priorityLevelName, 1)
		defer cleanup()

		highQPSClientName := "highqps-" + f.UniqueName
		lowQPSClientName := "lowqps-" + f.UniqueName
		framework.Logf("creating FlowSchema %q", flowSchemaName)
		_, cleanup = createFlowSchema(f, flowSchemaName, 1000, priorityLevelName, []string{highQPSClientName, lowQPSClientName})
		defer cleanup()

		ginkgo.By("waiting for testing flow schema and priority level to reach steady state")
		waitForSteadyState(f, flowSchemaName, priorityLevelName)

		type client struct {
			username                    string
			qps                         float64
			concurrencyMultiplier       float64 //lint:ignore U1000 field is actually used
			concurrency                 int32
			completedRequests           int32
			expectedCompletedPercentage float64 //lint:ignore U1000 field is actually used
		}
		clients := []client{
			{username: highQPSClientName, qps: 90, concurrencyMultiplier: 2.0, expectedCompletedPercentage: 0.90},
			{username: lowQPSClientName, qps: 4, concurrencyMultiplier: 0.5, expectedCompletedPercentage: 0.90},
		}

		framework.Logf("getting real concurrency")
		realConcurrency, err := getPriorityLevelConcurrency(f.ClientSet, priorityLevelName)
		framework.ExpectNoError(err)
		for i := range clients {
			clients[i].concurrency = int32(float64(realConcurrency) * clients[i].concurrencyMultiplier)
			if clients[i].concurrency < 1 {
				clients[i].concurrency = 1
			}
			framework.Logf("request concurrency for %q will be %d", clients[i].username, clients[i].concurrency)
		}

		ginkgo.By(fmt.Sprintf("starting uniform QPS load for %s", loadDuration.String()))
		var wg sync.WaitGroup
		for i := range clients {
			wg.Add(1)
			go func(c *client) {
				defer wg.Done()
				framework.Logf("starting uniform QPS load for %q: concurrency=%d, qps=%.1f", c.username, c.concurrency, c.qps)
				c.completedRequests = uniformQPSLoadConcurrent(f, c.username, c.concurrency, c.qps, loadDuration)
			}(&clients[i])
		}
		wg.Wait()

		ginkgo.By("checking completed requests with expected values")
		for _, client := range clients {
			// Each client should have 95% of its ideal number of completed requests.
			maxCompletedRequests := float64(client.concurrency) * client.qps * float64(loadDuration/time.Second)
			fractionCompleted := float64(client.completedRequests) / maxCompletedRequests
			framework.Logf("client %q completed %d/%d requests (%.1f%%)", client.username, client.completedRequests, int32(maxCompletedRequests), 100*fractionCompleted)
			if fractionCompleted < client.expectedCompletedPercentage {
				framework.Failf("client %q: got %.1f%% completed requests, want at least %.1f%%", client.username, 100*fractionCompleted, 100*client.expectedCompletedPercentage)
			}
		}
	})
})

// createPriorityLevel creates a priority level with the provided assured
// concurrency share.
func createPriorityLevel(f *framework.Framework, priorityLevelName string, assuredConcurrencyShares int32) (*flowcontrol.PriorityLevelConfiguration, func()) {
	createdPriorityLevel, err := f.ClientSet.FlowcontrolV1beta2().PriorityLevelConfigurations().Create(
		context.TODO(),
		&flowcontrol.PriorityLevelConfiguration{
			ObjectMeta: metav1.ObjectMeta{
				Name: priorityLevelName,
			},
			Spec: flowcontrol.PriorityLevelConfigurationSpec{
				Type: flowcontrol.PriorityLevelEnablementLimited,
				Limited: &flowcontrol.LimitedPriorityLevelConfiguration{
					AssuredConcurrencyShares: assuredConcurrencyShares,
					LimitResponse: flowcontrol.LimitResponse{
						Type: flowcontrol.LimitResponseTypeReject,
					},
				},
			},
		},
		metav1.CreateOptions{})
	framework.ExpectNoError(err)
	return createdPriorityLevel, func() {
		framework.ExpectNoError(f.ClientSet.FlowcontrolV1beta2().PriorityLevelConfigurations().Delete(context.TODO(), priorityLevelName, metav1.DeleteOptions{}))
	}
}

func getPriorityLevelConcurrency(c clientset.Interface, priorityLevelName string) (int32, error) {
	resp, err := c.CoreV1().RESTClient().Get().RequestURI("/metrics").DoRaw(context.TODO())
	if err != nil {
		return 0, err
	}
	sampleDecoder := expfmt.SampleDecoder{
		Dec:  expfmt.NewDecoder(bytes.NewBuffer(resp), expfmt.FmtText),
		Opts: &expfmt.DecodeOptions{},
	}
	for {
		var v model.Vector
		err := sampleDecoder.Decode(&v)
		if err != nil {
			if err == io.EOF {
				break
			}
			return 0, err
		}
		for _, metric := range v {
			if string(metric.Metric[model.MetricNameLabel]) != requestConcurrencyLimitMetricName {
				continue
			}
			if string(metric.Metric[priorityLevelLabelName]) != priorityLevelName {
				continue
			}
			return int32(metric.Value), nil
		}
	}
	return 0, errPriorityLevelNotFound
}

// createFlowSchema creates a flow schema referring to a particular priority
// level and matching the username provided.
func createFlowSchema(f *framework.Framework, flowSchemaName string, matchingPrecedence int32, priorityLevelName string, matchingUsernames []string) (*flowcontrol.FlowSchema, func()) {
	var subjects []flowcontrol.Subject
	for _, matchingUsername := range matchingUsernames {
		subjects = append(subjects, flowcontrol.Subject{
			Kind: flowcontrol.SubjectKindUser,
			User: &flowcontrol.UserSubject{
				Name: matchingUsername,
			},
		})
	}

	createdFlowSchema, err := f.ClientSet.FlowcontrolV1beta2().FlowSchemas().Create(
		context.TODO(),
		&flowcontrol.FlowSchema{
			ObjectMeta: metav1.ObjectMeta{
				Name: flowSchemaName,
			},
			Spec: flowcontrol.FlowSchemaSpec{
				MatchingPrecedence: matchingPrecedence,
				PriorityLevelConfiguration: flowcontrol.PriorityLevelConfigurationReference{
					Name: priorityLevelName,
				},
				DistinguisherMethod: &flowcontrol.FlowDistinguisherMethod{
					Type: flowcontrol.FlowDistinguisherMethodByUserType,
				},
				Rules: []flowcontrol.PolicyRulesWithSubjects{
					{
						Subjects: subjects,
						NonResourceRules: []flowcontrol.NonResourcePolicyRule{
							{
								Verbs:           []string{flowcontrol.VerbAll},
								NonResourceURLs: []string{flowcontrol.NonResourceAll},
							},
						},
					},
				},
			},
		},
		metav1.CreateOptions{})
	framework.ExpectNoError(err)
	return createdFlowSchema, func() {
		framework.ExpectNoError(f.ClientSet.FlowcontrolV1beta2().FlowSchemas().Delete(context.TODO(), flowSchemaName, metav1.DeleteOptions{}))
	}
}

// waitForSteadyState repeatedly polls the API server to check if the newly
// created flow schema and priority level have been seen by the APF controller
// by checking: (1) the dangling priority level reference condition in the flow
// schema status, and (2) metrics. The function times out after 30 seconds.
func waitForSteadyState(f *framework.Framework, flowSchemaName string, priorityLevelName string) {
	framework.ExpectNoError(wait.Poll(time.Second, 30*time.Second, func() (bool, error) {
		fs, err := f.ClientSet.FlowcontrolV1beta2().FlowSchemas().Get(context.TODO(), flowSchemaName, metav1.GetOptions{})
		if err != nil {
			return false, err
		}
		condition := apihelpers.GetFlowSchemaConditionByType(fs, flowcontrol.FlowSchemaConditionDangling)
		if condition == nil || condition.Status != flowcontrol.ConditionFalse {
			// The absence of the dangling status object implies that the APF
			// controller isn't done with syncing the flow schema object. And, of
			// course, the condition being anything but false means that steady state
			// hasn't been achieved.
			return false, nil
		}
		_, err = getPriorityLevelConcurrency(f.ClientSet, priorityLevelName)
		if err != nil {
			if err == errPriorityLevelNotFound {
				return false, nil
			}
			return false, err
		}
		return true, nil
	}))
}

// makeRequests creates a request to the API server and returns the response.
func makeRequest(f *framework.Framework, username string) *http.Response {
	config := f.ClientConfig()
	config.Impersonate.UserName = username
	config.RateLimiter = clientsideflowcontrol.NewFakeAlwaysRateLimiter()
	config.Impersonate.Groups = []string{"system:authenticated"}
	roundTripper, err := rest.TransportFor(config)
	framework.ExpectNoError(err)

	req, err := http.NewRequest(http.MethodGet, f.ClientSet.CoreV1().RESTClient().Get().AbsPath("version").URL().String(), nil)
	framework.ExpectNoError(err)

	response, err := roundTripper.RoundTrip(req)
	framework.ExpectNoError(err)
	return response
}

func getPriorityLevelUID(response *http.Response) string {
	return response.Header.Get(flowcontrol.ResponseHeaderMatchedPriorityLevelConfigurationUID)
}

func getFlowSchemaUID(response *http.Response) string {
	return response.Header.Get(flowcontrol.ResponseHeaderMatchedFlowSchemaUID)
}

// uniformQPSLoadSingle loads the API server with requests at a uniform <qps>
// for <loadDuration> time. The number of successfully completed requests is
// returned.
func uniformQPSLoadSingle(f *framework.Framework, username string, qps float64, loadDuration time.Duration) int32 {
	var completed int32
	var wg sync.WaitGroup
	ticker := time.NewTicker(time.Duration(float64(time.Second) / qps))
	defer ticker.Stop()
	timer := time.NewTimer(loadDuration)
	for {
		select {
		case <-ticker.C:
			wg.Add(1)
			// Each request will have a non-zero latency. In addition, there may be
			// multiple concurrent requests in-flight. As a result, a request may
			// take longer than the time between two different consecutive ticks
			// regardless of whether a requests is accepted or rejected. For example,
			// in cases with clients making requests far above their concurrency
			// share, with little time between consecutive requests, due to limited
			// concurrency, newer requests will be enqueued until older ones
			// complete. Hence the synchronisation with sync.WaitGroup.
			go func() {
				defer wg.Done()
				makeRequest(f, username)
				atomic.AddInt32(&completed, 1)
			}()
		case <-timer.C:
			// Still in-flight requests should not contribute to the completed count.
			totalCompleted := atomic.LoadInt32(&completed)
			wg.Wait() // do not leak goroutines
			return totalCompleted
		}
	}
}

// uniformQPSLoadConcurrent loads the API server with a <concurrency> number of
// clients impersonating to be <username>, each creating requests at a uniform
// rate defined by <qps>. The sum of number of successfully completed requests
// across all concurrent clients is returned.
func uniformQPSLoadConcurrent(f *framework.Framework, username string, concurrency int32, qps float64, loadDuration time.Duration) int32 {
	var completed int32
	var wg sync.WaitGroup
	wg.Add(int(concurrency))
	for i := int32(0); i < concurrency; i++ {
		go func() {
			defer wg.Done()
			atomic.AddInt32(&completed, uniformQPSLoadSingle(f, username, qps, loadDuration))
		}()
	}
	wg.Wait()
	return completed
}
