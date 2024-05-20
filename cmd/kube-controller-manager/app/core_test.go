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

package app

import (
	"context"
	"testing"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/discovery"
	fakediscovery "k8s.io/client-go/discovery/fake"
	"k8s.io/client-go/informers"
	clientset "k8s.io/client-go/kubernetes"
	fakeclientset "k8s.io/client-go/kubernetes/fake"
	restclient "k8s.io/client-go/rest"
	"k8s.io/klog/v2"
	"k8s.io/klog/v2/ktesting"
)

// TestClientBuilder inherits ClientBuilder and can accept a given fake clientset.
type TestClientBuilder struct {
	clientset clientset.Interface
}

func (TestClientBuilder) Config(logger klog.Logger, name string) (*restclient.Config, error) {
	return nil, nil
}
func (TestClientBuilder) ConfigOrDie(logger klog.Logger, name string) *restclient.Config {
	return &restclient.Config{}
}

func (TestClientBuilder) Client(logger klog.Logger, name string) (clientset.Interface, error) {
	return nil, nil
}
func (m TestClientBuilder) ClientOrDie(logger klog.Logger, name string) clientset.Interface {
	return m.clientset
}

func (m TestClientBuilder) DiscoveryClient(logger klog.Logger, name string) (discovery.DiscoveryInterface, error) {
	return m.clientset.Discovery(), nil
}
func (m TestClientBuilder) DiscoveryClientOrDie(logger klog.Logger, name string) discovery.DiscoveryInterface {
	ret, err := m.DiscoveryClient(logger, name)
	if err != nil {
		panic(err)
	}
	return ret
}

// FakeDiscoveryWithError inherits DiscoveryInterface(via FakeDiscovery) with some methods accepting testing data.
type FakeDiscoveryWithError struct {
	fakediscovery.FakeDiscovery
	PossibleResources []*metav1.APIResourceList
	Err               error
}

func (d FakeDiscoveryWithError) ServerPreferredNamespacedResources() ([]*metav1.APIResourceList, error) {
	return d.PossibleResources, d.Err
}

// FakeDiscoveryWithError inherits Clientset(via FakeClientset) with overridden Discovery method.
type FakeClientSet struct {
	fakeclientset.Clientset
	DiscoveryObj *FakeDiscoveryWithError
}

func (c *FakeClientSet) Discovery() discovery.DiscoveryInterface {
	return c.DiscoveryObj
}

func (c *FakeClientSet) GetPossibleResources() []*metav1.APIResourceList {
	return c.DiscoveryObj.PossibleResources
}

// Create a fake Clientset with its Discovery method overridden.
func NewFakeClientset(fakeDiscovery FakeDiscoveryWithError) *FakeClientSet {
	cs := &FakeClientSet{}
	cs.DiscoveryObj = &fakeDiscovery
	return cs
}

func possibleDiscoveryResource() []*metav1.APIResourceList {
	return []*metav1.APIResourceList{
		{
			GroupVersion: "create/v1",
			APIResources: []metav1.APIResource{
				{
					Name:       "jobs",
					Verbs:      []string{"create", "list", "watch", "delete"},
					ShortNames: []string{"jz"},
					Categories: []string{"all"},
				},
			},
		},
	}
}

func TestController_DiscoveryError(t *testing.T) {
	controllerDescriptorMap := map[string]*ControllerDescriptor{
		"ResourceQuotaController":          newResourceQuotaControllerDescriptor(),
		"GarbageCollectorController":       newGarbageCollectorControllerDescriptor(),
		"EndpointSliceController":          newEndpointSliceControllerDescriptor(),
		"EndpointSliceMirroringController": newEndpointSliceMirroringControllerDescriptor(),
		"PodDisruptionBudgetController":    newDisruptionControllerDescriptor(),
	}

	tcs := map[string]struct {
		discoveryError    error
		expectedErr       bool
		possibleResources []*metav1.APIResourceList
	}{
		"No Discovery Error": {
			discoveryError:    nil,
			possibleResources: possibleDiscoveryResource(),
			expectedErr:       false,
		},
		"Discovery Calls Partially Failed": {
			discoveryError:    new(discovery.ErrGroupDiscoveryFailed),
			possibleResources: possibleDiscoveryResource(),
			expectedErr:       false,
		},
	}
	for name, test := range tcs {
		testDiscovery := FakeDiscoveryWithError{Err: test.discoveryError, PossibleResources: test.possibleResources}
		testClientset := NewFakeClientset(testDiscovery)
		testClientBuilder := TestClientBuilder{clientset: testClientset}
		testInformerFactory := informers.NewSharedInformerFactoryWithOptions(testClientset, time.Duration(1))
		ctx := ControllerContext{
			ClientBuilder:                   testClientBuilder,
			InformerFactory:                 testInformerFactory,
			ObjectOrMetadataInformerFactory: testInformerFactory,
			InformersStarted:                make(chan struct{}),
		}
		for controllerName, controllerDesc := range controllerDescriptorMap {
			_, _, err := controllerDesc.GetInitFunc()(context.TODO(), ctx, controllerName)
			if test.expectedErr != (err != nil) {
				t.Errorf("%v test failed for use case: %v", controllerName, name)
			}
		}
		logger, _ := ktesting.NewTestContext(t)
		_, _, err := startModifiedNamespaceController(
			context.TODO(), ctx, testClientset, testClientBuilder.ConfigOrDie(logger, "namespace-controller"))
		if test.expectedErr != (err != nil) {
			t.Errorf("Namespace Controller test failed for use case: %v", name)
		}
	}
}
