/*
Copyright 2023 The Kubernetes Authors.

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

package defaultservicecidr

import (
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	networkingapiv1alpha1 "k8s.io/api/networking/v1alpha1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes/fake"
	k8stesting "k8s.io/client-go/testing"
	"k8s.io/client-go/tools/record"
	"k8s.io/utils/ptr"
)

const (
	defaultIPv4CIDR = "10.16.0.0/16"
	defaultIPv6CIDR = "2001:db8::/64"
)

func newController(t *testing.T, objects []*networkingapiv1alpha1.ServiceCIDR) (*fake.Clientset, *Controller) {
	client := fake.NewSimpleClientset()

	informerFactory := informers.NewSharedInformerFactory(client, 0)
	serviceCIDRInformer := informerFactory.Networking().V1alpha1().ServiceCIDRs()

	store := serviceCIDRInformer.Informer().GetStore()
	for _, obj := range objects {
		err := store.Add(obj)
		if err != nil {
			t.Fatal(err)
		}

	}
	c := &Controller{
		client:             client,
		interval:           time.Second,
		cidrs:              []string{defaultIPv4CIDR, defaultIPv6CIDR},
		eventRecorder:      record.NewFakeRecorder(100),
		serviceCIDRLister:  serviceCIDRInformer.Lister(),
		serviceCIDRsSynced: func() bool { return true },
	}

	return client, c
}

func TestControllerSync(t *testing.T) {
	testCases := []struct {
		name    string
		cidrs   []*networkingapiv1alpha1.ServiceCIDR
		actions [][]string // verb and resource
	}{
		{
			name:    "no existing service CIDRs",
			actions: [][]string{{"create", "servicecidrs"}, {"patch", "servicecidrs"}},
		},
		{
			name: "existing default service CIDR update Ready condition",
			cidrs: []*networkingapiv1alpha1.ServiceCIDR{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name: DefaultServiceCIDRName,
					},
					Spec: networkingapiv1alpha1.ServiceCIDRSpec{
						CIDRs: []string{defaultIPv4CIDR, defaultIPv6CIDR},
					},
				},
			},
			actions: [][]string{{"patch", "servicecidrs"}},
		},
		{
			name: "existing default service CIDR not matching cidrs",
			cidrs: []*networkingapiv1alpha1.ServiceCIDR{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name: DefaultServiceCIDRName,
					},
					Spec: networkingapiv1alpha1.ServiceCIDRSpec{
						CIDRs: []string{"fd00::/112"},
					},
				},
			},
		},
		{
			name: "existing default service CIDR not ready",
			cidrs: []*networkingapiv1alpha1.ServiceCIDR{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name: DefaultServiceCIDRName,
					},
					Spec: networkingapiv1alpha1.ServiceCIDRSpec{
						CIDRs: []string{defaultIPv4CIDR, defaultIPv6CIDR},
					},
					Status: networkingapiv1alpha1.ServiceCIDRStatus{
						Conditions: []metav1.Condition{
							{
								Type:   string(networkingapiv1alpha1.ServiceCIDRConditionReady),
								Status: metav1.ConditionFalse,
							},
						},
					},
				},
			},
		},
		{
			name: "existing default service CIDR being deleted",
			cidrs: []*networkingapiv1alpha1.ServiceCIDR{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name:              DefaultServiceCIDRName,
						DeletionTimestamp: ptr.To(metav1.Now()),
					},
					Spec: networkingapiv1alpha1.ServiceCIDRSpec{
						CIDRs: []string{defaultIPv4CIDR, defaultIPv6CIDR},
					},
				},
			},
		},
		{
			name: "existing service CIDRs but not default",
			cidrs: []*networkingapiv1alpha1.ServiceCIDR{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name: "non-default-cidr",
					},
					Spec: networkingapiv1alpha1.ServiceCIDRSpec{
						CIDRs: []string{defaultIPv4CIDR, defaultIPv6CIDR},
					},
				},
			},
			actions: [][]string{{"create", "servicecidrs"}, {"patch", "servicecidrs"}},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			client, controller := newController(t, tc.cidrs)
			controller.sync()
			expectAction(t, client.Actions(), tc.actions)
		})
	}
}

func expectAction(t *testing.T, actions []k8stesting.Action, expected [][]string) {
	t.Helper()
	if len(actions) != len(expected) {
		t.Fatalf("Expected at least %d actions, got %d \ndiff: %v", len(expected), len(actions), cmp.Diff(expected, actions))
	}

	for i, action := range actions {
		verb := expected[i][0]
		if action.GetVerb() != verb {
			t.Errorf("Expected action %d verb to be %s, got %s", i, verb, action.GetVerb())
		}
		resource := expected[i][1]
		if action.GetResource().Resource != resource {
			t.Errorf("Expected action %d resource to be %s, got %s", i, resource, action.GetResource().Resource)
		}
	}
}
