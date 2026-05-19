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
	networkingapiv1 "k8s.io/api/networking/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
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

func newController(t *testing.T, cidrsFromFlags []string, objects ...*networkingapiv1.ServiceCIDR) (*fake.Clientset, *Controller) {
	var runtimeObjects []runtime.Object
	for _, cidr := range objects {
		runtimeObjects = append(runtimeObjects, cidr)
	}

	client := fake.NewSimpleClientset(runtimeObjects...)

	informerFactory := informers.NewSharedInformerFactory(client, 0)
	serviceCIDRInformer := informerFactory.Networking().V1().ServiceCIDRs()

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
		cidrs:              cidrsFromFlags,
		eventRecorder:      record.NewFakeRecorder(100),
		serviceCIDRLister:  serviceCIDRInformer.Lister(),
		serviceCIDRsSynced: func() bool { return true },
	}

	return client, c
}

func TestControllerSync(t *testing.T) {
	testCases := []struct {
		name    string
		cidrs   []*networkingapiv1.ServiceCIDR
		actions [][]string // verb and resource
	}{
		{
			name:    "no existing service CIDRs",
			actions: [][]string{{"create", "servicecidrs"}, {"patch", "servicecidrs"}},
		},
		{
			name: "existing default service CIDR update Ready condition",
			cidrs: []*networkingapiv1.ServiceCIDR{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name: DefaultServiceCIDRName,
					},
					Spec: networkingapiv1.ServiceCIDRSpec{
						CIDRs: []string{defaultIPv4CIDR, defaultIPv6CIDR},
					},
				},
			},
			actions: [][]string{{"patch", "servicecidrs"}},
		},
		{
			name: "existing default service CIDR not matching cidrs",
			cidrs: []*networkingapiv1.ServiceCIDR{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name: DefaultServiceCIDRName,
					},
					Spec: networkingapiv1.ServiceCIDRSpec{
						CIDRs: []string{"fd00::/112"},
					},
				},
			},
		},
		{
			name: "existing default service CIDR not ready",
			cidrs: []*networkingapiv1.ServiceCIDR{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name: DefaultServiceCIDRName,
					},
					Spec: networkingapiv1.ServiceCIDRSpec{
						CIDRs: []string{defaultIPv4CIDR, defaultIPv6CIDR},
					},
					Status: networkingapiv1.ServiceCIDRStatus{
						Conditions: []metav1.Condition{
							{
								Type:   string(networkingapiv1.ServiceCIDRConditionReady),
								Status: metav1.ConditionFalse,
							},
						},
					},
				},
			},
		},
		{
			name: "existing default service CIDR being deleted",
			cidrs: []*networkingapiv1.ServiceCIDR{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name:              DefaultServiceCIDRName,
						DeletionTimestamp: ptr.To(metav1.Now()),
					},
					Spec: networkingapiv1.ServiceCIDRSpec{
						CIDRs: []string{defaultIPv4CIDR, defaultIPv6CIDR},
					},
				},
			},
		},
		{
			name: "existing service CIDRs but not default",
			cidrs: []*networkingapiv1.ServiceCIDR{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name: "non-default-cidr",
					},
					Spec: networkingapiv1.ServiceCIDRSpec{
						CIDRs: []string{defaultIPv4CIDR, defaultIPv6CIDR},
					},
				},
			},
			actions: [][]string{{"create", "servicecidrs"}, {"patch", "servicecidrs"}},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			client, controller := newController(t, []string{defaultIPv4CIDR, defaultIPv6CIDR}, tc.cidrs...)
			controller.sync()
			expectAction(t, client.Actions(), tc.actions)
		})
	}
}

func TestControllerSyncConversions(t *testing.T) {
	testCases := []struct {
		name            string
		controllerCIDRs []string
		existingCIDR    *networkingapiv1.ServiceCIDR
		expectedAction  [][]string // verb, resource, [subresource]
	}{
		{
			name:            "flags match ServiceCIDRs",
			controllerCIDRs: []string{defaultIPv4CIDR, defaultIPv6CIDR},
			existingCIDR: &networkingapiv1.ServiceCIDR{
				ObjectMeta: metav1.ObjectMeta{
					Name: DefaultServiceCIDRName,
				},
				Spec: networkingapiv1.ServiceCIDRSpec{
					CIDRs: []string{defaultIPv4CIDR, defaultIPv6CIDR},
				},
				Status: networkingapiv1.ServiceCIDRStatus{}, // No conditions
			},
			expectedAction: [][]string{{"patch", "servicecidrs", "status"}},
		},
		{
			name:            "existing Ready=False condition, cidrs match -> no patch",
			controllerCIDRs: []string{defaultIPv4CIDR, defaultIPv6CIDR},
			existingCIDR: &networkingapiv1.ServiceCIDR{
				ObjectMeta: metav1.ObjectMeta{
					Name: DefaultServiceCIDRName,
				},
				Spec: networkingapiv1.ServiceCIDRSpec{
					CIDRs: []string{defaultIPv4CIDR, defaultIPv6CIDR},
				},
				Status: networkingapiv1.ServiceCIDRStatus{
					Conditions: []metav1.Condition{
						{
							Type:   networkingapiv1.ServiceCIDRConditionReady,
							Status: metav1.ConditionFalse,
							Reason: "SomeReason",
						},
					},
				},
			},
			expectedAction: [][]string{}, // No patch expected, just logs/events
		},
		{
			name:            "existing Ready=True condition -> no patch",
			controllerCIDRs: []string{defaultIPv4CIDR, defaultIPv6CIDR},
			existingCIDR: &networkingapiv1.ServiceCIDR{
				ObjectMeta: metav1.ObjectMeta{
					Name: DefaultServiceCIDRName,
				},
				Spec: networkingapiv1.ServiceCIDRSpec{
					CIDRs: []string{defaultIPv4CIDR, defaultIPv6CIDR},
				},
				Status: networkingapiv1.ServiceCIDRStatus{
					Conditions: []metav1.Condition{
						{
							Type:   networkingapiv1.ServiceCIDRConditionReady,
							Status: metav1.ConditionTrue,
						},
					},
				},
			},
			expectedAction: [][]string{},
		},
		{
			name:            "ServiceCIDR being deleted -> no patch",
			controllerCIDRs: []string{defaultIPv4CIDR, defaultIPv6CIDR},
			existingCIDR: &networkingapiv1.ServiceCIDR{
				ObjectMeta: metav1.ObjectMeta{
					Name:              DefaultServiceCIDRName,
					DeletionTimestamp: ptr.To(metav1.Now()),
				},
				Spec: networkingapiv1.ServiceCIDRSpec{
					CIDRs: []string{defaultIPv4CIDR, defaultIPv6CIDR},
				},
				Status: networkingapiv1.ServiceCIDRStatus{},
			},
			expectedAction: [][]string{},
		},
		{
			name:            "IPv4 to IPv4 IPv6 is ok",
			controllerCIDRs: []string{defaultIPv4CIDR, defaultIPv6CIDR},
			existingCIDR: &networkingapiv1.ServiceCIDR{
				ObjectMeta: metav1.ObjectMeta{
					Name: DefaultServiceCIDRName,
				},
				Spec: networkingapiv1.ServiceCIDRSpec{
					CIDRs: []string{defaultIPv4CIDR}, // Existing has both
				},
				Status: networkingapiv1.ServiceCIDRStatus{},
			},
			expectedAction: [][]string{{"update", "servicecidrs"}},
		},
		{
			name:            "IPv4 to IPv6 IPv4 - switching primary IP family leaves in inconsistent state",
			controllerCIDRs: []string{defaultIPv6CIDR, defaultIPv4CIDR},
			existingCIDR: &networkingapiv1.ServiceCIDR{
				ObjectMeta: metav1.ObjectMeta{
					Name: DefaultServiceCIDRName,
				},
				Spec: networkingapiv1.ServiceCIDRSpec{
					CIDRs: []string{defaultIPv4CIDR}, // Existing has both
				},
				Status: networkingapiv1.ServiceCIDRStatus{},
			},
			expectedAction: [][]string{},
		},
		{
			name:            "IPv6 to IPv6 IPv4",
			controllerCIDRs: []string{defaultIPv6CIDR, defaultIPv4CIDR},
			existingCIDR: &networkingapiv1.ServiceCIDR{
				ObjectMeta: metav1.ObjectMeta{
					Name: DefaultServiceCIDRName,
				},
				Spec: networkingapiv1.ServiceCIDRSpec{
					CIDRs: []string{defaultIPv6CIDR}, // Existing has both
				},
				Status: networkingapiv1.ServiceCIDRStatus{},
			},
			expectedAction: [][]string{{"update", "servicecidrs"}},
		},
		{
			name:            "IPv6 to IPv4 IPv6 - switching primary IP family leaves in inconsistent state",
			controllerCIDRs: []string{defaultIPv4CIDR, defaultIPv6CIDR},
			existingCIDR: &networkingapiv1.ServiceCIDR{
				ObjectMeta: metav1.ObjectMeta{
					Name: DefaultServiceCIDRName,
				},
				Spec: networkingapiv1.ServiceCIDRSpec{
					CIDRs: []string{defaultIPv6CIDR}, // Existing has both
				},
				Status: networkingapiv1.ServiceCIDRStatus{},
			},
			expectedAction: [][]string{},
		},
		{
			name:            "IPv6 IPv4 to IPv4 IPv6 - switching primary IP family leaves in inconsistent state",
			controllerCIDRs: []string{defaultIPv4CIDR, defaultIPv6CIDR},
			existingCIDR: &networkingapiv1.ServiceCIDR{
				ObjectMeta: metav1.ObjectMeta{
					Name: DefaultServiceCIDRName,
				},
				Spec: networkingapiv1.ServiceCIDRSpec{
					CIDRs: []string{defaultIPv6CIDR, defaultIPv4CIDR}, // Existing has both
				},
				Status: networkingapiv1.ServiceCIDRStatus{},
			},
			expectedAction: [][]string{},
		},
		{
			name:            "IPv4 IPv6 to IPv4 - needs operator attention for the IPv6 remaining Services",
			controllerCIDRs: []string{defaultIPv4CIDR},
			existingCIDR: &networkingapiv1.ServiceCIDR{
				ObjectMeta: metav1.ObjectMeta{
					Name: DefaultServiceCIDRName,
				},
				Spec: networkingapiv1.ServiceCIDRSpec{
					CIDRs: []string{defaultIPv4CIDR, defaultIPv6CIDR},
				},
				Status: networkingapiv1.ServiceCIDRStatus{},
			},
			expectedAction: [][]string{},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// Initialize controller and client with the existing ServiceCIDR
			client, controller := newController(t, tc.controllerCIDRs, tc.existingCIDR)

			// Call the syncStatus method directly
			err := controller.sync()
			if err != nil {
				t.Errorf("unexpected error: %v", err)
			}

			// Verify the expected actions
			expectAction(t, client.Actions(), tc.expectedAction)
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
