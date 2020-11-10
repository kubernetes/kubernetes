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
	"fmt"
	"io"
	"net/http"
	"sync"
	"sync/atomic"
	"time"

	"github.com/onsi/ginkgo"
	"github.com/prometheus/common/expfmt"
	"github.com/prometheus/common/model"

	flowcontrolv1alpha1 "k8s.io/api/flowcontrol/v1alpha1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/rest"
	"k8s.io/kubernetes/test/e2e/framework"
)

const (
	requestConcurrencyLimitMetricName      = "apiserver_flowcontrol_request_concurrency_limit"
	requestConcurrencyLimitMetricLabelName = "priorityLevel"
)

var _ = SIGDescribe("[Feature:APIPriorityAndFairness]", func() {
	f := framework.NewDefaultFramework("flowschemas")

	ginkgo.It("should ensure that requests can be classified by testing flow-schemas/priority-levels", func() {
		testingFlowSchemaName := "e2e-testing-flowschema"
		testingPriorityLevelName := "e2e-testing-prioritylevel"
		matchingUsername := "noxu"
		nonMatchingUsername := "foo"

		ginkgo.By("creating a testing prioritylevel")
		createdPriorityLevel, err := f.ClientSet.FlowcontrolV1alpha1().PriorityLevelConfigurations().Create(
			context.TODO(),
			&flowcontrolv1alpha1.PriorityLevelConfiguration{
				ObjectMeta: metav1.ObjectMeta{
					Name: testingPriorityLevelName,
				},
				Spec: flowcontrolv1alpha1.PriorityLevelConfigurationSpec{
					Type: flowcontrolv1alpha1.PriorityLevelEnablementLimited,
					Limited: &flowcontrolv1alpha1.LimitedPriorityLevelConfiguration{
						AssuredConcurrencyShares: 1, // will have at minimum 1 concurrency share
						LimitResponse: flowcontrolv1alpha1.LimitResponse{
							Type: flowcontrolv1alpha1.LimitResponseTypeReject,
						},
					},
				},
			},
			metav1.CreateOptions{})
		framework.ExpectNoError(err)

		defer func() {
			// clean-ups
			err := f.ClientSet.FlowcontrolV1alpha1().PriorityLevelConfigurations().Delete(context.TODO(), testingPriorityLevelName, metav1.DeleteOptions{})
			framework.ExpectNoError(err)
			err = f.ClientSet.FlowcontrolV1alpha1().FlowSchemas().Delete(context.TODO(), testingFlowSchemaName, metav1.DeleteOptions{})
			framework.ExpectNoError(err)
		}()

		ginkgo.By("creating a testing flowschema")
		createdFlowSchema, err := f.ClientSet.FlowcontrolV1alpha1().FlowSchemas().Create(
			context.TODO(),
			&flowcontrolv1alpha1.FlowSchema{
				ObjectMeta: metav1.ObjectMeta{
					Name: testingFlowSchemaName,
				},
				Spec: flowcontrolv1alpha1.FlowSchemaSpec{
					MatchingPrecedence: 1000, // a rather higher precedence to ensure it make effect
					PriorityLevelConfiguration: flowcontrolv1alpha1.PriorityLevelConfigurationReference{
						Name: testingPriorityLevelName,
					},
					DistinguisherMethod: &flowcontrolv1alpha1.FlowDistinguisherMethod{
						Type: flowcontrolv1alpha1.FlowDistinguisherMethodByUserType,
					},
					Rules: []flowcontrolv1alpha1.PolicyRulesWithSubjects{
						{
							Subjects: []flowcontrolv1alpha1.Subject{
								{
									Kind: flowcontrolv1alpha1.SubjectKindUser,
									User: &flowcontrolv1alpha1.UserSubject{
										Name: matchingUsername,
									},
								},
							},
							NonResourceRules: []flowcontrolv1alpha1.NonResourcePolicyRule{
								{
									Verbs:           []string{flowcontrolv1alpha1.VerbAll},
									NonResourceURLs: []string{flowcontrolv1alpha1.NonResourceAll},
								},
							},
						},
					},
				},
			},
			metav1.CreateOptions{})
		framework.ExpectNoError(err)

		ginkgo.By("response headers should contain flow-schema/priority-level uid")

		if !testResponseHeaderMatches(f, matchingUsername, string(createdPriorityLevel.UID), string(createdFlowSchema.UID)) {
			framework.Failf("matching user doesnt received UID for the testing priority-level and flow-schema")
		}
		if testResponseHeaderMatches(f, nonMatchingUsername, string(createdPriorityLevel.UID), string(createdPriorityLevel.UID)) {
			framework.Failf("non-matching user unexpectedly received UID for the testing priority-level and flow-schema")
		}
	})

	ginkgo.It("should ensure that requests can't be drowned out", func() {
		flowSchemaNamePrefix := "e2e-testing-flowschema"
		priorityLevelNamePrefix := "e2e-testing-prioritylevel"
		loadDuration := 10 * time.Second
		type client struct {
			username              string
			qps                   float64
			priorityLevelName     string
			concurrencyMultiplier float64
			concurrency           int32
			flowSchemaName        string
			matchingPrecedence    int32
			completedRequests     int32
		}
		clients := []client{
			// "elephant" refers to a client that creates requests at a much higher
			// QPS than its counter-part and well above its concurrency share limit.
			// In contrast, the mouse stays under its concurrency shares.
			// Additionally, the "elephant" client also has a higher matching
			// precedence for its flow schema.
			{username: "elephant", qps: 100.0, concurrencyMultiplier: 2.0, matchingPrecedence: 999},
			{username: "mouse", qps: 5.0, concurrencyMultiplier: 0.5, matchingPrecedence: 1000},
		}

		ginkgo.By("creating test priority levels and flow schemas")
		for i := range clients {
			clients[i].priorityLevelName = fmt.Sprintf("%s-%s", priorityLevelNamePrefix, clients[i].username)
			framework.Logf("creating PriorityLevel %q", clients[i].priorityLevelName)
			_, err := f.ClientSet.FlowcontrolV1alpha1().PriorityLevelConfigurations().Create(
				context.TODO(),
				&flowcontrolv1alpha1.PriorityLevelConfiguration{
					ObjectMeta: metav1.ObjectMeta{
						Name: clients[i].priorityLevelName,
					},
					Spec: flowcontrolv1alpha1.PriorityLevelConfigurationSpec{
						Type: flowcontrolv1alpha1.PriorityLevelEnablementLimited,
						Limited: &flowcontrolv1alpha1.LimitedPriorityLevelConfiguration{
							AssuredConcurrencyShares: 1,
							LimitResponse: flowcontrolv1alpha1.LimitResponse{
								Type: flowcontrolv1alpha1.LimitResponseTypeReject,
							},
						},
					},
				},
				metav1.CreateOptions{})
			framework.ExpectNoError(err)
			defer func(name string) {
				framework.ExpectNoError(f.ClientSet.FlowcontrolV1alpha1().PriorityLevelConfigurations().Delete(context.TODO(), name, metav1.DeleteOptions{}))
			}(clients[i].priorityLevelName)
			clients[i].flowSchemaName = fmt.Sprintf("%s-%s", flowSchemaNamePrefix, clients[i].username)
			framework.Logf("creating FlowSchema %q", clients[i].flowSchemaName)
			_, err = f.ClientSet.FlowcontrolV1alpha1().FlowSchemas().Create(
				context.TODO(),
				&flowcontrolv1alpha1.FlowSchema{
					ObjectMeta: metav1.ObjectMeta{
						Name: clients[i].flowSchemaName,
					},
					Spec: flowcontrolv1alpha1.FlowSchemaSpec{
						MatchingPrecedence: clients[i].matchingPrecedence,
						PriorityLevelConfiguration: flowcontrolv1alpha1.PriorityLevelConfigurationReference{
							Name: clients[i].priorityLevelName,
						},
						DistinguisherMethod: &flowcontrolv1alpha1.FlowDistinguisherMethod{
							Type: flowcontrolv1alpha1.FlowDistinguisherMethodByUserType,
						},
						Rules: []flowcontrolv1alpha1.PolicyRulesWithSubjects{
							{
								Subjects: []flowcontrolv1alpha1.Subject{
									{
										Kind: flowcontrolv1alpha1.SubjectKindUser,
										User: &flowcontrolv1alpha1.UserSubject{
											Name: clients[i].username,
										},
									},
								},
								NonResourceRules: []flowcontrolv1alpha1.NonResourcePolicyRule{
									{
										Verbs:           []string{flowcontrolv1alpha1.VerbAll},
										NonResourceURLs: []string{flowcontrolv1alpha1.NonResourceAll},
									},
								},
							},
						},
					},
				},
				metav1.CreateOptions{})
			framework.ExpectNoError(err)
			defer func(name string) {
				framework.ExpectNoError(f.ClientSet.FlowcontrolV1alpha1().FlowSchemas().Delete(context.TODO(), name, metav1.DeleteOptions{}))
			}(clients[i].flowSchemaName)
		}

		ginkgo.By("getting request concurrency from metrics")
		for i := range clients {
			resp, err := f.ClientSet.CoreV1().RESTClient().Get().RequestURI("/metrics").DoRaw(context.TODO())
			framework.ExpectNoError(err)
			sampleDecoder := expfmt.SampleDecoder{
				Dec:  expfmt.NewDecoder(bytes.NewBuffer(resp), expfmt.FmtText),
				Opts: &expfmt.DecodeOptions{},
			}
			for {
				var v model.Vector
				err := sampleDecoder.Decode(&v)
				if err == io.EOF {
					break
				}
				framework.ExpectNoError(err)
				for _, metric := range v {
					if string(metric.Metric[model.MetricNameLabel]) != requestConcurrencyLimitMetricName {
						continue
					}
					if string(metric.Metric[requestConcurrencyLimitMetricLabelName]) != clients[i].priorityLevelName {
						continue
					}
					clients[i].concurrency = int32(float64(metric.Value) * clients[i].concurrencyMultiplier)
					if clients[i].concurrency < 1 {
						clients[i].concurrency = 1
					}
					framework.Logf("request concurrency for %q will be %d (concurrency share = %d)", clients[i].username, clients[i].concurrency, int32(metric.Value))
				}
			}
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
			if fractionCompleted < 0.95 {
				framework.Failf("client %q: got %.1f%% completed requests, want at least 95%%", client.username, 100*fractionCompleted)
			}
		}
	})
})

// makeRequests creates a request to the API server and returns the response.
func makeRequest(f *framework.Framework, username string) *http.Response {
	config := rest.CopyConfig(f.ClientConfig())
	config.Impersonate.UserName = username
	roundTripper, err := rest.TransportFor(config)
	framework.ExpectNoError(err)

	req, err := http.NewRequest(http.MethodGet, f.ClientSet.CoreV1().RESTClient().Get().AbsPath("version").URL().String(), nil)
	framework.ExpectNoError(err)

	response, err := roundTripper.RoundTrip(req)
	framework.ExpectNoError(err)
	return response
}

func testResponseHeaderMatches(f *framework.Framework, impersonatingUser, plUID, fsUID string) bool {
	response := makeRequest(f, impersonatingUser)
	if response.Header.Get(flowcontrolv1alpha1.ResponseHeaderMatchedFlowSchemaUID) != fsUID {
		return false
	}
	if response.Header.Get(flowcontrolv1alpha1.ResponseHeaderMatchedPriorityLevelConfigurationUID) != plUID {
		return false
	}
	return true
}

// uniformQPSLoadSingle loads the API server with requests at a uniform <qps>
// for <loadDuration> time. The number of successfully completed requests is
// returned.
func uniformQPSLoadSingle(f *framework.Framework, username string, qps float64, loadDuration time.Duration) int32 {
	var completed int32
	var wg sync.WaitGroup
	ticker := time.NewTicker(time.Duration(1e9/qps) * time.Nanosecond)
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
	return atomic.LoadInt32(&completed)
}
