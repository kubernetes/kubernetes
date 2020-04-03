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
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/prometheus/common/expfmt"
	"github.com/prometheus/common/model"

	flowcontrolv1alpha1 "k8s.io/api/flowcontrol/v1alpha1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/wait"
	genericfeatures "k8s.io/apiserver/pkg/features"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/kubernetes/pkg/master"
	"k8s.io/kubernetes/test/integration/framework"
)

const (
	sharedConcurrencyMetricsName      = "apiserver_flowcontrol_request_concurrency_limit"
	dispatchedRequestCountMetricsName = "apiserver_flowcontrol_dispatched_requests_total"
	labelPriorityLevel                = "priorityLevel"
	timeout                           = time.Second * 10
)

func setup(t testing.TB) (*httptest.Server, *rest.Config, framework.CloseFunc) {
	opts := framework.MasterConfigOptions{EtcdOptions: framework.DefaultEtcdOptions()}
	opts.EtcdOptions.DefaultStorageMediaType = "application/vnd.kubernetes.protobuf"
	masterConfig := framework.NewIntegrationTestMasterConfigWithOptions(&opts)
	resourceConfig := master.DefaultAPIResourceConfigSource()
	resourceConfig.EnableVersions(schema.GroupVersion{
		Group:   "flowcontrol.apiserver.k8s.io",
		Version: "v1alpha1",
	})
	masterConfig.GenericConfig.MaxRequestsInFlight = 1
	masterConfig.GenericConfig.MaxMutatingRequestsInFlight = 1
	masterConfig.GenericConfig.OpenAPIConfig = framework.DefaultOpenAPIConfig()
	masterConfig.ExtraConfig.APIResourceConfigSource = resourceConfig
	_, s, closeFn := framework.RunAMaster(masterConfig)

	return s, masterConfig.GenericConfig.LoopbackClientConfig, closeFn
}

func TestPriorityLevelIsolation(t *testing.T) {
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, genericfeatures.APIPriorityAndFairness, true)()
	// NOTE: disabling the feature should fail the test
	_, loopbackConfig, closeFn := setup(t)
	defer closeFn()

	loopbackClient := clientset.NewForConfigOrDie(loopbackConfig)
	noxu1Client := getClientFor(loopbackConfig, "noxu1")
	noxu2Client := getClientFor(loopbackConfig, "noxu2")

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
	defer close(stopCh)

	// "elephant"
	streamRequests(concurrencyShares+queueLength, func() {
		_, err := noxu1Client.CoreV1().Namespaces().List(context.Background(), metav1.ListOptions{})
		if err != nil {
			t.Error(err)
		}
	}, stopCh)
	// "mouse"
	streamRequests(3, func() {
		_, err := noxu2Client.CoreV1().Namespaces().List(context.Background(), metav1.ListOptions{})
		if err != nil {
			t.Error(err)
		}
	}, stopCh)

	time.Sleep(time.Second * 10) // running in background for a while

	reqCounts, err := getRequestCountOfPriorityLevel(loopbackClient)
	if err != nil {
		t.Error(err)
	}

	noxu1RequestCount := reqCounts[priorityLevelNoxu1.Name]
	noxu2RequestCount := reqCounts[priorityLevelNoxu2.Name]

	// Theoretically, the actual expected value of request counts upon the two priority-level should be
	// the equal. We're deliberately lax to make flakes super rare.
	if (noxu1RequestCount/2) > noxu2RequestCount || (noxu2RequestCount/2) > noxu1RequestCount {
		t.Errorf("imbalanced requests made by noxu1/2: (%d:%d)", noxu1RequestCount, noxu2RequestCount)
	}
}

func getClientFor(loopbackConfig *rest.Config, username string) clientset.Interface {
	config := &rest.Config{
		Host:        loopbackConfig.Host,
		QPS:         -1,
		BearerToken: loopbackConfig.BearerToken,
		Impersonate: rest.ImpersonationConfig{
			UserName: username,
		},
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

func getRequestCountOfPriorityLevel(c clientset.Interface) (map[string]int, error) {
	resp, err := getMetrics(c)
	if err != nil {
		return nil, err
	}

	dec := expfmt.NewDecoder(strings.NewReader(string(resp)), expfmt.FmtText)
	decoder := expfmt.SampleDecoder{
		Dec:  dec,
		Opts: &expfmt.DecodeOptions{},
	}

	reqCounts := make(map[string]int)
	for {
		var v model.Vector
		if err := decoder.Decode(&v); err != nil {
			if err == io.EOF {
				// Expected loop termination condition.
				return reqCounts, nil
			}
			return nil, fmt.Errorf("failed decoding metrics: %v", err)
		}
		for _, metric := range v {
			switch name := string(metric.Metric[model.MetricNameLabel]); name {
			case dispatchedRequestCountMetricsName:
				reqCounts[string(metric.Metric[labelPriorityLevel])] = int(metric.Value)
			}
		}
	}
}

func createPriorityLevelAndBindingFlowSchemaForUser(c clientset.Interface, username string, concurrencyShares, queuelength int) (*flowcontrolv1alpha1.PriorityLevelConfiguration, *flowcontrolv1alpha1.FlowSchema, error) {
	pl, err := c.FlowcontrolV1alpha1().PriorityLevelConfigurations().Create(context.Background(), &flowcontrolv1alpha1.PriorityLevelConfiguration{
		ObjectMeta: metav1.ObjectMeta{
			Name: username,
		},
		Spec: flowcontrolv1alpha1.PriorityLevelConfigurationSpec{
			Type: flowcontrolv1alpha1.PriorityLevelEnablementLimited,
			Limited: &flowcontrolv1alpha1.LimitedPriorityLevelConfiguration{
				AssuredConcurrencyShares: int32(concurrencyShares),
				LimitResponse: flowcontrolv1alpha1.LimitResponse{
					Type: flowcontrolv1alpha1.LimitResponseTypeQueue,
					Queuing: &flowcontrolv1alpha1.QueuingConfiguration{
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
	fs, err := c.FlowcontrolV1alpha1().FlowSchemas().Create(context.TODO(), &flowcontrolv1alpha1.FlowSchema{
		ObjectMeta: metav1.ObjectMeta{
			Name: username,
		},
		Spec: flowcontrolv1alpha1.FlowSchemaSpec{
			DistinguisherMethod: &flowcontrolv1alpha1.FlowDistinguisherMethod{
				Type: flowcontrolv1alpha1.FlowDistinguisherMethodByUserType,
			},
			MatchingPrecedence: 1000,
			PriorityLevelConfiguration: flowcontrolv1alpha1.PriorityLevelConfigurationReference{
				Name: username,
			},
			Rules: []flowcontrolv1alpha1.PolicyRulesWithSubjects{
				{
					ResourceRules: []flowcontrolv1alpha1.ResourcePolicyRule{
						{
							Verbs:        []string{flowcontrolv1alpha1.VerbAll},
							APIGroups:    []string{flowcontrolv1alpha1.APIGroupAll},
							Resources:    []string{flowcontrolv1alpha1.ResourceAll},
							Namespaces:   []string{flowcontrolv1alpha1.NamespaceEvery},
							ClusterScope: true,
						},
					},
					Subjects: []flowcontrolv1alpha1.Subject{
						{
							Kind: flowcontrolv1alpha1.SubjectKindUser,
							User: &flowcontrolv1alpha1.UserSubject{
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
		fs, err := c.FlowcontrolV1alpha1().FlowSchemas().Get(context.TODO(), username, metav1.GetOptions{})
		if err != nil {
			return false, err
		}
		for _, condition := range fs.Status.Conditions {
			if condition.Type == flowcontrolv1alpha1.FlowSchemaConditionDangling {
				if condition.Status == flowcontrolv1alpha1.ConditionFalse {
					return true, nil
				}
			}
		}
		return false, nil
	})
}

func streamRequests(parallel int, request func(), stopCh <-chan struct{}) {
	for i := 0; i < parallel; i++ {
		go func() {
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
