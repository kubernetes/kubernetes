/*
Copyright 2017 The Kubernetes Authors.

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

package thirdparty

import (
	"reflect"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/util/workqueue"
	"k8s.io/kube-aggregator/pkg/apis/apiregistration"
	"k8s.io/kubernetes/pkg/apis/extensions"
	listers "k8s.io/kubernetes/pkg/client/listers/extensions/internalversion"
)

func TestEnqueue(t *testing.T) {
	c := tprRegistrationController{
		queue: workqueue.NewNamedRateLimitingQueue(workqueue.DefaultControllerRateLimiter(), "tpr-autoregister"),
	}

	tpr := &extensions.ThirdPartyResource{
		ObjectMeta: metav1.ObjectMeta{Name: "resource.group.example.com"},
		Versions: []extensions.APIVersion{
			{Name: "v1alpha1"},
			{Name: "v1"},
		},
	}
	c.enqueueTPR(tpr)

	first, _ := c.queue.Get()
	expectedFirst := schema.GroupVersion{Group: "group.example.com", Version: "v1alpha1"}
	if first != expectedFirst {
		t.Errorf("expected %v, got %v", expectedFirst, first)
	}

	second, _ := c.queue.Get()
	expectedSecond := schema.GroupVersion{Group: "group.example.com", Version: "v1"}
	if second != expectedSecond {
		t.Errorf("expected %v, got %v", expectedSecond, second)
	}
}

func TestHandleTPR(t *testing.T) {
	tests := []struct {
		name         string
		startingTPRs []*extensions.ThirdPartyResource
		version      schema.GroupVersion

		expectedAdded   []*apiregistration.APIService
		expectedRemoved []string
	}{
		{
			name: "simple add",
			startingTPRs: []*extensions.ThirdPartyResource{
				{
					ObjectMeta: metav1.ObjectMeta{Name: "resource.group.com"},
					Versions: []extensions.APIVersion{
						{Name: "v1"},
					},
				},
			},
			version: schema.GroupVersion{Group: "group.com", Version: "v1"},

			expectedAdded: []*apiregistration.APIService{
				{
					ObjectMeta: metav1.ObjectMeta{Name: "v1.group.com"},
					Spec: apiregistration.APIServiceSpec{
						Group:    "group.com",
						Version:  "v1",
						Priority: 500,
					},
				},
			},
		},
		{
			name: "simple remove",
			startingTPRs: []*extensions.ThirdPartyResource{
				{
					ObjectMeta: metav1.ObjectMeta{Name: "resource.group.com"},
					Versions: []extensions.APIVersion{
						{Name: "v1"},
					},
				},
			},
			version: schema.GroupVersion{Group: "group.com", Version: "v2"},

			expectedRemoved: []string{"v2.group.com"},
		},
	}

	for _, test := range tests {
		registration := &fakeAPIServiceRegistration{}
		tprCache := cache.NewIndexer(cache.DeletionHandlingMetaNamespaceKeyFunc, cache.Indexers{cache.NamespaceIndex: cache.MetaNamespaceIndexFunc})
		tprLister := listers.NewThirdPartyResourceLister(tprCache)
		c := tprRegistrationController{
			tprLister:              tprLister,
			apiServiceRegistration: registration,
		}
		for i := range test.startingTPRs {
			tprCache.Add(test.startingTPRs[i])
		}

		c.handleTPR(test.version)

		if !reflect.DeepEqual(test.expectedAdded, registration.added) {
			t.Errorf("%s expected %v, got %v", test.name, test.expectedAdded, registration.added)
		}
		if !reflect.DeepEqual(test.expectedRemoved, registration.removed) {
			t.Errorf("%s expected %v, got %v", test.name, test.expectedRemoved, registration.removed)
		}
	}

}

type fakeAPIServiceRegistration struct {
	added   []*apiregistration.APIService
	removed []string
}

func (a *fakeAPIServiceRegistration) AddAPIServiceToSync(in *apiregistration.APIService) {
	a.added = append(a.added, in)
}
func (a *fakeAPIServiceRegistration) RemoveAPIServiceToSync(name string) {
	a.removed = append(a.removed, name)
}
