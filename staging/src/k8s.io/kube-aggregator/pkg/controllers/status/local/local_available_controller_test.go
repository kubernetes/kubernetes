/*
Copyright 2024 The Kubernetes Authors.

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

package external

import (
	"strings"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/dump"
	clienttesting "k8s.io/client-go/testing"
	"k8s.io/client-go/tools/cache"
	apiregistration "k8s.io/kube-aggregator/pkg/apis/apiregistration/v1"
	"k8s.io/kube-aggregator/pkg/client/clientset_generated/clientset/fake"
	apiregistrationclient "k8s.io/kube-aggregator/pkg/client/clientset_generated/clientset/typed/apiregistration/v1"
	listers "k8s.io/kube-aggregator/pkg/client/listers/apiregistration/v1"
	availabilitymetrics "k8s.io/kube-aggregator/pkg/controllers/status/metrics"
	"k8s.io/utils/ptr"
)

const (
	testServicePort int32 = 1234
)

func newLocalAPIService(name string) *apiregistration.APIService {
	return &apiregistration.APIService{
		ObjectMeta: metav1.ObjectMeta{Name: name},
	}
}

func newRemoteAPIService(name string) *apiregistration.APIService {
	return &apiregistration.APIService{
		ObjectMeta: metav1.ObjectMeta{Name: name},
		Spec: apiregistration.APIServiceSpec{
			Group:   strings.SplitN(name, ".", 2)[0],
			Version: strings.SplitN(name, ".", 2)[1],
			Service: &apiregistration.ServiceReference{
				Namespace: "foo",
				Name:      "bar",
				Port:      ptr.To(testServicePort),
			},
		},
	}
}

func TestSync(t *testing.T) {
	tests := []struct {
		name string

		apiServiceName string
		apiServices    []runtime.Object

		expectedAvailability apiregistration.APIServiceCondition
		expectedAction       bool
	}{
		{
			name:           "local",
			apiServiceName: "local.group",
			apiServices:    []runtime.Object{newLocalAPIService("local.group")},
			expectedAvailability: apiregistration.APIServiceCondition{
				Type:    apiregistration.Available,
				Status:  apiregistration.ConditionTrue,
				Reason:  "Local",
				Message: "Local APIServices are always available",
			},
			expectedAction: true,
		},
		{
			name:           "remote",
			apiServiceName: "remote.group",
			apiServices:    []runtime.Object{newRemoteAPIService("remote.group")},
			expectedAction: false,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			fakeClient := fake.NewSimpleClientset(tc.apiServices...)
			apiServiceIndexer := cache.NewIndexer(cache.MetaNamespaceKeyFunc, cache.Indexers{cache.NamespaceIndex: cache.MetaNamespaceIndexFunc})
			for _, obj := range tc.apiServices {
				if err := apiServiceIndexer.Add(obj); err != nil {
					t.Fatalf("failed to add object to indexer: %v", err)
				}
			}

			c := AvailableConditionController{
				apiServiceClient: fakeClient.ApiregistrationV1(),
				apiServiceLister: listers.NewAPIServiceLister(apiServiceIndexer),
				metrics:          availabilitymetrics.New(),
			}
			if err := c.sync(tc.apiServiceName); err != nil {
				t.Fatalf("unexpect sync error: %v", err)
			}

			// ought to have one action writing status
			if e, a := tc.expectedAction, len(fakeClient.Actions()) == 1; e != a {
				t.Fatalf("%v expected %v, got %v", tc.name, e, fakeClient.Actions())
			}
			if tc.expectedAction {
				action, ok := fakeClient.Actions()[0].(clienttesting.UpdateAction)
				if !ok {
					t.Fatalf("%v got %v", tc.name, ok)
				}

				if e, a := 1, len(action.GetObject().(*apiregistration.APIService).Status.Conditions); e != a {
					t.Fatalf("%v expected %v, got %v", tc.name, e, action.GetObject())
				}
				condition := action.GetObject().(*apiregistration.APIService).Status.Conditions[0]
				if e, a := tc.expectedAvailability.Type, condition.Type; e != a {
					t.Errorf("%v expected %v, got %#v", tc.name, e, condition)
				}
				if e, a := tc.expectedAvailability.Status, condition.Status; e != a {
					t.Errorf("%v expected %v, got %#v", tc.name, e, condition)
				}
				if e, a := tc.expectedAvailability.Reason, condition.Reason; e != a {
					t.Errorf("%v expected %v, got %#v", tc.name, e, condition)
				}
				if e, a := tc.expectedAvailability.Message, condition.Message; !strings.HasPrefix(a, e) {
					t.Errorf("%v expected %v, got %#v", tc.name, e, condition)
				}
				if condition.LastTransitionTime.IsZero() {
					t.Error("expected lastTransitionTime to be non-zero")
				}
			}
		})
	}
}

func TestUpdateAPIServiceStatus(t *testing.T) {
	foo := &apiregistration.APIService{Status: apiregistration.APIServiceStatus{Conditions: []apiregistration.APIServiceCondition{{Type: "foo"}}}}
	bar := &apiregistration.APIService{Status: apiregistration.APIServiceStatus{Conditions: []apiregistration.APIServiceCondition{{Type: "bar"}}}}

	fakeClient := fake.NewSimpleClientset(foo)
	c := AvailableConditionController{
		apiServiceClient: fakeClient.ApiregistrationV1().(apiregistrationclient.APIServicesGetter),
		metrics:          availabilitymetrics.New(),
	}

	if _, err := c.updateAPIServiceStatus(foo, foo); err != nil {
		t.Fatalf("unexpected updateAPIServiceStatus error: %v", err)
	}
	if e, a := 0, len(fakeClient.Actions()); e != a {
		t.Error(dump.Pretty(fakeClient.Actions()))
	}

	fakeClient.ClearActions()
	if _, err := c.updateAPIServiceStatus(foo, bar); err != nil {
		t.Fatalf("unexpected updateAPIServiceStatus error: %v", err)
	}
	if e, a := 1, len(fakeClient.Actions()); e != a {
		t.Error(dump.Pretty(fakeClient.Actions()))
	}
}
