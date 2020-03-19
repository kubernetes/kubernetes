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
	"sync"
	"testing"
	"time"

	"github.com/prometheus/common/expfmt"
	"github.com/prometheus/common/model"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

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
	dispatchedRequestCountMetricsName               = "apiserver_flowcontrol_dispatched_requests_total"
	dispatchedRequestCountMetricsLabelPriorityLevel = "priorityLevel"
	timeout                                         = time.Second * 10
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
	masterConfig.GenericConfig.MaxRequestsInFlight = 5
	masterConfig.GenericConfig.MaxMutatingRequestsInFlight = 5
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

	priorityLevelNoxu1, _, err := createPriorityLevelAndBindingFlowSchemaForUser(loopbackClient, "noxu1")
	require.NoError(t, err)
	priorityLevelNoxu2, _, err := createPriorityLevelAndBindingFlowSchemaForUser(loopbackClient, "noxu2")
	require.NoError(t, err)

	wg := &sync.WaitGroup{}
	// "elephant"
	streamRequests(wg, 10, 100, func() {
		_, err := noxu1Client.CoreV1().Namespaces().List(context.TODO(), metav1.ListOptions{})
		require.NoError(t, err)
	})

	streamRequests(nil, 1, 100, func() {
		_, err := noxu2Client.CoreV1().Namespaces().List(context.TODO(), metav1.ListOptions{})
		require.NoError(t, err)
	})

	wg.Wait()

	dispatchedCountNoxu1, err := getRequestCountOfPriorityLevel(loopbackClient, priorityLevelNoxu1.Name)
	require.NoError(t, err)
	dispatchedCountNoxu2, err := getRequestCountOfPriorityLevel(loopbackClient, priorityLevelNoxu2.Name)
	require.NoError(t, err)

	assert.Equal(t, 1000, dispatchedCountNoxu1)
	assert.Equal(t, 100, dispatchedCountNoxu2)
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

func getRequestCountOfPriorityLevel(c clientset.Interface, priorityLevelName string) (int, error) {
	resp, err := c.CoreV1().
		RESTClient().
		Get().
		RequestURI("/metrics").
		DoRaw(context.TODO())
	if err != nil {
		return 0, err
	}

	dec := expfmt.NewDecoder(strings.NewReader(string(resp)), expfmt.FmtText)
	decoder := expfmt.SampleDecoder{
		Dec:  dec,
		Opts: &expfmt.DecodeOptions{},
	}

	for {
		var v model.Vector
		if err := decoder.Decode(&v); err != nil {
			if err == io.EOF {
				// Expected loop termination condition.
				return 0, fmt.Errorf("no dispatched-count metrics found for priorityLevel %v", priorityLevelName)
			}
			return 0, fmt.Errorf("failed decoding metrics: %v", err)
		}
		for _, metric := range v {
			switch name := string(metric.Metric[model.MetricNameLabel]); name {
			case dispatchedRequestCountMetricsName:
				if priorityLevelName == string(metric.Metric[dispatchedRequestCountMetricsLabelPriorityLevel]) {
					return int(metric.Value), nil
				}
			}
		}
	}
}

func createPriorityLevelAndBindingFlowSchemaForUser(c clientset.Interface, username string) (*flowcontrolv1alpha1.PriorityLevelConfiguration, *flowcontrolv1alpha1.FlowSchema, error) {
	pl, err := c.FlowcontrolV1alpha1().PriorityLevelConfigurations().Create(context.TODO(), &flowcontrolv1alpha1.PriorityLevelConfiguration{
		ObjectMeta: metav1.ObjectMeta{
			Name: username,
		},
		Spec: flowcontrolv1alpha1.PriorityLevelConfigurationSpec{
			Type: flowcontrolv1alpha1.PriorityLevelEnablementLimited,
			Limited: &flowcontrolv1alpha1.LimitedPriorityLevelConfiguration{
				AssuredConcurrencyShares: 10,
				LimitResponse: flowcontrolv1alpha1.LimitResponse{
					Type: flowcontrolv1alpha1.LimitResponseTypeQueue,
					Queuing: &flowcontrolv1alpha1.QueuingConfiguration{
						Queues:           100,
						HandSize:         1,
						QueueLengthLimit: 10,
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

func streamRequests(wg *sync.WaitGroup, parallel, times int, request func()) {
	for i := 0; i < parallel; i++ {
		if wg != nil {
			wg.Add(1)
		}
		go func() {
			for j := 0; j < times; j++ {
				request()
			}
			if wg != nil {
				wg.Done()
			}
		}()
	}
}
