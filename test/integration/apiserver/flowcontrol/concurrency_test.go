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
	"fmt"
	"io"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/prometheus/common/expfmt"
	"github.com/prometheus/common/model"

	flowcontrol "k8s.io/api/flowcontrol/v1beta2"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	genericfeatures "k8s.io/apiserver/pkg/features"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/kubernetes/cmd/kube-apiserver/app/options"
	"k8s.io/kubernetes/test/integration/framework"
)

const (
	sharedConcurrencyMetricsName      = "apiserver_flowcontrol_request_concurrency_limit"
	dispatchedRequestCountMetricsName = "apiserver_flowcontrol_dispatched_requests_total"
	rejectedRequestCountMetricsName   = "apiserver_flowcontrol_rejected_requests_total"
	labelPriorityLevel                = "priority_level"
	timeout                           = time.Second * 10
)

func setup(t testing.TB, maxReadonlyRequestsInFlight, MaxMutatingRequestsInFlight int) (*rest.Config, framework.TearDownFunc) {
	_, kubeConfig, tearDownFn := framework.StartTestServer(t, framework.TestServerSetup{
		ModifyServerRunOptions: func(opts *options.ServerRunOptions) {
			// Ensure all clients are allowed to send requests.
			opts.Authorization.Modes = []string{"AlwaysAllow"}
			opts.GenericServerRunOptions.MaxRequestsInFlight = maxReadonlyRequestsInFlight
			opts.GenericServerRunOptions.MaxMutatingRequestsInFlight = MaxMutatingRequestsInFlight
		},
	})
	return kubeConfig, tearDownFn
}

func TestPriorityLevelIsolation(t *testing.T) {
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, genericfeatures.APIPriorityAndFairness, true)()
	// NOTE: disabling the feature should fail the test
	kubeConfig, closeFn := setup(t, 1, 1)
	defer closeFn()

	loopbackClient := clientset.NewForConfigOrDie(kubeConfig)
	noxu1Client := getClientFor(kubeConfig, "noxu1")
	noxu2Client := getClientFor(kubeConfig, "noxu2")

	queueLength := 50
	concurrencyShares := 1

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

	sharedConcurrency, err := getSharedConcurrencyOfPriorityLevel(loopbackClient)
	if err != nil {
		t.Error(err)
	}

	if 1 != sharedConcurrency[priorityLevelNoxu1.Name] {
		t.Errorf("unexpected shared concurrency %v instead of %v", sharedConcurrency[priorityLevelNoxu1.Name], 1)
	}
	if 1 != sharedConcurrency[priorityLevelNoxu2.Name] {
		t.Errorf("unexpected shared concurrency %v instead of %v", sharedConcurrency[priorityLevelNoxu2.Name], 1)
	}

	stopCh := make(chan struct{})
	wg := sync.WaitGroup{}
	defer func() {
		close(stopCh)
		wg.Wait()
	}()

	// "elephant"
	wg.Add(concurrencyShares + queueLength)
	streamRequests(concurrencyShares+queueLength, func() {
		_, err := noxu1Client.CoreV1().Namespaces().List(context.Background(), metav1.ListOptions{})
		if err != nil {
			t.Error(err)
		}
	}, &wg, stopCh)
	// "mouse"
	wg.Add(3)
	streamRequests(3, func() {
		_, err := noxu2Client.CoreV1().Namespaces().List(context.Background(), metav1.ListOptions{})
		if err != nil {
			t.Error(err)
		}
	}, &wg, stopCh)

	time.Sleep(time.Second * 10) // running in background for a while

	allDispatchedReqCounts, rejectedReqCounts, err := getRequestCountOfPriorityLevel(loopbackClient)
	if err != nil {
		t.Error(err)
	}

	noxu1RequestCount := allDispatchedReqCounts[priorityLevelNoxu1.Name]
	noxu2RequestCount := allDispatchedReqCounts[priorityLevelNoxu2.Name]

	if rejectedReqCounts[priorityLevelNoxu1.Name] > 0 {
		t.Errorf(`%v requests from the "elephant" stream were rejected unexpectedly`, rejectedReqCounts[priorityLevelNoxu2.Name])
	}
	if rejectedReqCounts[priorityLevelNoxu2.Name] > 0 {
		t.Errorf(`%v requests from the "mouse" stream were rejected unexpectedly`, rejectedReqCounts[priorityLevelNoxu2.Name])
	}

	// Theoretically, the actual expected value of request counts upon the two priority-level should be
	// the equal. We're deliberately lax to make flakes super rare.
	if (noxu1RequestCount/2) > noxu2RequestCount || (noxu2RequestCount/2) > noxu1RequestCount {
		t.Errorf("imbalanced requests made by noxu1/2: (%d:%d)", noxu1RequestCount, noxu2RequestCount)
	}
}

func getClientFor(loopbackConfig *rest.Config, username string) clientset.Interface {
	config := rest.CopyConfig(loopbackConfig)
	config.Impersonate = rest.ImpersonationConfig{
		UserName: username,
	}
	return clientset.NewForConfigOrDie(config)
}

func getMetrics(c clientset.Interface) (string, error) {
	resp, err := c.CoreV1().
		RESTClient().
		Get().
		RequestURI("/metrics").
		DoRaw(context.Background())
	if err != nil {
		return "", err
	}
	return string(resp), err
}

func getSharedConcurrencyOfPriorityLevel(c clientset.Interface) (map[string]int, error) {
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
			case sharedConcurrencyMetricsName:
				concurrency[string(metric.Metric[labelPriorityLevel])] = int(metric.Value)
			}
		}
	}
}

func getRequestCountOfPriorityLevel(c clientset.Interface) (map[string]int, map[string]int, error) {
	resp, err := getMetrics(c)
	if err != nil {
		return nil, nil, err
	}

	dec := expfmt.NewDecoder(strings.NewReader(string(resp)), expfmt.FmtText)
	decoder := expfmt.SampleDecoder{
		Dec:  dec,
		Opts: &expfmt.DecodeOptions{},
	}

	allReqCounts := make(map[string]int)
	rejectReqCounts := make(map[string]int)
	for {
		var v model.Vector
		if err := decoder.Decode(&v); err != nil {
			if err == io.EOF {
				// Expected loop termination condition.
				return allReqCounts, rejectReqCounts, nil
			}
			return nil, nil, fmt.Errorf("failed decoding metrics: %v", err)
		}
		for _, metric := range v {
			switch name := string(metric.Metric[model.MetricNameLabel]); name {
			case dispatchedRequestCountMetricsName:
				allReqCounts[string(metric.Metric[labelPriorityLevel])] = int(metric.Value)
			case rejectedRequestCountMetricsName:
				rejectReqCounts[string(metric.Metric[labelPriorityLevel])] = int(metric.Value)
			}
		}
	}
}

func createPriorityLevelAndBindingFlowSchemaForUser(c clientset.Interface, username string, concurrencyShares, queuelength int) (*flowcontrol.PriorityLevelConfiguration, *flowcontrol.FlowSchema, error) {
	pl, err := c.FlowcontrolV1beta2().PriorityLevelConfigurations().Create(context.Background(), &flowcontrol.PriorityLevelConfiguration{
		ObjectMeta: metav1.ObjectMeta{
			Name: username,
		},
		Spec: flowcontrol.PriorityLevelConfigurationSpec{
			Type: flowcontrol.PriorityLevelEnablementLimited,
			Limited: &flowcontrol.LimitedPriorityLevelConfiguration{
				AssuredConcurrencyShares: int32(concurrencyShares),
				LimitResponse: flowcontrol.LimitResponse{
					Type: flowcontrol.LimitResponseTypeQueue,
					Queuing: &flowcontrol.QueuingConfiguration{
						Queues:           100,
						HandSize:         1,
						QueueLengthLimit: int32(queuelength),
					},
				},
			},
		},
	}, metav1.CreateOptions{})
	if err != nil {
		return nil, nil, err
	}
	fs, err := c.FlowcontrolV1beta2().FlowSchemas().Create(context.TODO(), &flowcontrol.FlowSchema{
		ObjectMeta: metav1.ObjectMeta{
			Name: username,
		},
		Spec: flowcontrol.FlowSchemaSpec{
			DistinguisherMethod: &flowcontrol.FlowDistinguisherMethod{
				Type: flowcontrol.FlowDistinguisherMethodByUserType,
			},
			MatchingPrecedence: 1000,
			PriorityLevelConfiguration: flowcontrol.PriorityLevelConfigurationReference{
				Name: username,
			},
			Rules: []flowcontrol.PolicyRulesWithSubjects{
				{
					ResourceRules: []flowcontrol.ResourcePolicyRule{
						{
							Verbs:        []string{flowcontrol.VerbAll},
							APIGroups:    []string{flowcontrol.APIGroupAll},
							Resources:    []string{flowcontrol.ResourceAll},
							Namespaces:   []string{flowcontrol.NamespaceEvery},
							ClusterScope: true,
						},
					},
					Subjects: []flowcontrol.Subject{
						{
							Kind: flowcontrol.SubjectKindUser,
							User: &flowcontrol.UserSubject{
								Name: username,
							},
						},
					},
				},
			},
		},
	}, metav1.CreateOptions{})
	if err != nil {
		return nil, nil, err
	}

	return pl, fs, wait.Poll(time.Second, timeout, func() (bool, error) {
		fs, err := c.FlowcontrolV1beta2().FlowSchemas().Get(context.TODO(), username, metav1.GetOptions{})
		if err != nil {
			return false, err
		}
		for _, condition := range fs.Status.Conditions {
			if condition.Type == flowcontrol.FlowSchemaConditionDangling {
				if condition.Status == flowcontrol.ConditionFalse {
					return true, nil
				}
			}
		}
		return false, nil
	})
}

func streamRequests(parallel int, request func(), wg *sync.WaitGroup, stopCh <-chan struct{}) {
	for i := 0; i < parallel; i++ {
		go func() {
			defer wg.Done()
			for {
				select {
				case <-stopCh:
					return
				default:
					request()
				}
			}
		}()
	}
}
