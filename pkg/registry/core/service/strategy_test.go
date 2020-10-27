/*
Copyright 2014 The Kubernetes Authors.

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

package service

import (
	"net"
	"reflect"
	"testing"

	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/diff"
	"k8s.io/apimachinery/pkg/util/intstr"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/registry/rest"
	api "k8s.io/kubernetes/pkg/apis/core"
	_ "k8s.io/kubernetes/pkg/apis/core/install"

	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/kubernetes/pkg/features"
)

func newStrategy(cidr string, hasSecondary bool) (testStrategy Strategy, testStatusStrategy Strategy) {
	_, testCIDR, err := net.ParseCIDR(cidr)
	if err != nil {
		panic("invalid CIDR")
	}
	testStrategy, _ = StrategyForServiceCIDRs(*testCIDR, hasSecondary)
	testStatusStrategy = NewServiceStatusStrategy(testStrategy)
	return
}

func TestExportService(t *testing.T) {
	testStrategy, _ := newStrategy("10.0.0.0/16", false)
	tests := []struct {
		objIn     runtime.Object
		objOut    runtime.Object
		exact     bool
		expectErr bool
	}{
		{
			objIn: &api.Service{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "foo",
					Namespace: "bar",
				},
				Status: api.ServiceStatus{
					LoadBalancer: api.LoadBalancerStatus{
						Ingress: []api.LoadBalancerIngress{
							{IP: "1.2.3.4"},
						},
					},
				},
			},
			objOut: &api.Service{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "foo",
					Namespace: "bar",
				},
			},
			exact: true,
		},
		{
			objIn: &api.Service{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "foo",
					Namespace: "bar",
				},
				Spec: api.ServiceSpec{
					ClusterIPs: []string{"10.0.0.1"},
				},
				Status: api.ServiceStatus{
					LoadBalancer: api.LoadBalancerStatus{
						Ingress: []api.LoadBalancerIngress{
							{IP: "1.2.3.4"},
						},
					},
				},
			},
			objOut: &api.Service{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "foo",
					Namespace: "bar",
				},
				Spec: api.ServiceSpec{
					ClusterIPs: nil,
				},
			},
		},
		{
			objIn: &api.Service{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "foo",
					Namespace: "bar",
				},
				Spec: api.ServiceSpec{
					ClusterIPs: []string{"10.0.0.1", "2001::1"},
				},
				Status: api.ServiceStatus{
					LoadBalancer: api.LoadBalancerStatus{
						Ingress: []api.LoadBalancerIngress{
							{IP: "1.2.3.4"},
						},
					},
				},
			},
			objOut: &api.Service{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "foo",
					Namespace: "bar",
				},
				Spec: api.ServiceSpec{
					ClusterIPs: nil,
				},
			},
		},

		{
			objIn:     &api.Pod{},
			expectErr: true,
		},
	}

	for _, test := range tests {
		err := testStrategy.Export(genericapirequest.NewContext(), test.objIn, test.exact)
		if err != nil {
			if !test.expectErr {
				t.Errorf("unexpected error: %v", err)
			}
			continue
		}
		if test.expectErr {
			t.Error("unexpected non-error")
			continue
		}
		if !reflect.DeepEqual(test.objIn, test.objOut) {
			t.Errorf("expected:\n%v\nsaw:\n%v\n", test.objOut, test.objIn)
		}
	}
}

func TestCheckGeneratedNameError(t *testing.T) {
	testStrategy, _ := newStrategy("10.0.0.0/16", false)
	expect := errors.NewNotFound(api.Resource("foos"), "bar")
	if err := rest.CheckGeneratedNameError(testStrategy, expect, &api.Service{}); err != expect {
		t.Errorf("NotFoundError should be ignored: %v", err)
	}

	expect = errors.NewAlreadyExists(api.Resource("foos"), "bar")
	if err := rest.CheckGeneratedNameError(testStrategy, expect, &api.Service{}); err != expect {
		t.Errorf("AlreadyExists should be returned when no GenerateName field: %v", err)
	}

	expect = errors.NewAlreadyExists(api.Resource("foos"), "bar")
	if err := rest.CheckGeneratedNameError(testStrategy, expect, &api.Service{ObjectMeta: metav1.ObjectMeta{GenerateName: "foo"}}); err == nil || !errors.IsServerTimeout(err) {
		t.Errorf("expected try again later error: %v", err)
	}
}

func makeValidService() api.Service {
	return api.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name:            "valid",
			Namespace:       "default",
			Labels:          map[string]string{},
			Annotations:     map[string]string{},
			ResourceVersion: "1",
		},
		Spec: api.ServiceSpec{
			Selector:        map[string]string{"key": "val"},
			SessionAffinity: "None",
			Type:            api.ServiceTypeClusterIP,
			Ports:           []api.ServicePort{{Name: "p", Protocol: "TCP", Port: 8675, TargetPort: intstr.FromInt(8675)}},
		},
	}
}

// TODO: This should be done on types that are not part of our API
func TestBeforeUpdate(t *testing.T) {
	testCases := []struct {
		name      string
		tweakSvc  func(oldSvc, newSvc *api.Service) // given basic valid services, each test case can customize them
		expectErr bool
	}{
		{
			name: "no change",
			tweakSvc: func(oldSvc, newSvc *api.Service) {
				// nothing
			},
			expectErr: false,
		},
		{
			name: "change port",
			tweakSvc: func(oldSvc, newSvc *api.Service) {
				newSvc.Spec.Ports[0].Port++
			},
			expectErr: false,
		},
		{
			name: "bad namespace",
			tweakSvc: func(oldSvc, newSvc *api.Service) {
				newSvc.Namespace = "#$%%invalid"
			},
			expectErr: true,
		},
		{
			name: "change name",
			tweakSvc: func(oldSvc, newSvc *api.Service) {
				newSvc.Name += "2"
			},
			expectErr: true,
		},
		{
			name: "change ClusterIP",
			tweakSvc: func(oldSvc, newSvc *api.Service) {
				oldSvc.Spec.ClusterIPs = []string{"1.2.3.4"}
				newSvc.Spec.ClusterIPs = []string{"4.3.2.1"}
			},
			expectErr: true,
		},
		{
			name: "change selector",
			tweakSvc: func(oldSvc, newSvc *api.Service) {
				newSvc.Spec.Selector = map[string]string{"newkey": "newvalue"}
			},
			expectErr: false,
		},
	}

	for _, tc := range testCases {
		strategy, _ := newStrategy("172.30.0.0/16", false)

		oldSvc := makeValidService()
		newSvc := makeValidService()
		tc.tweakSvc(&oldSvc, &newSvc)
		ctx := genericapirequest.NewDefaultContext()
		err := rest.BeforeUpdate(strategy, ctx, runtime.Object(&oldSvc), runtime.Object(&newSvc))
		if tc.expectErr && err == nil {
			t.Errorf("unexpected non-error for %q", tc.name)
		}
		if !tc.expectErr && err != nil {
			t.Errorf("unexpected error for %q: %v", tc.name, err)
		}
	}
}

func TestServiceStatusStrategy(t *testing.T) {
	_, testStatusStrategy := newStrategy("10.0.0.0/16", false)
	ctx := genericapirequest.NewDefaultContext()
	if !testStatusStrategy.NamespaceScoped() {
		t.Errorf("Service must be namespace scoped")
	}
	oldService := makeValidService()
	newService := makeValidService()
	oldService.ResourceVersion = "4"
	newService.ResourceVersion = "4"
	newService.Spec.SessionAffinity = "ClientIP"
	newService.Status = api.ServiceStatus{
		LoadBalancer: api.LoadBalancerStatus{
			Ingress: []api.LoadBalancerIngress{
				{IP: "127.0.0.2"},
			},
		},
	}
	testStatusStrategy.PrepareForUpdate(ctx, &newService, &oldService)
	if newService.Status.LoadBalancer.Ingress[0].IP != "127.0.0.2" {
		t.Errorf("Service status updates should allow change of status fields")
	}
	if newService.Spec.SessionAffinity != "None" {
		t.Errorf("PrepareForUpdate should have preserved old spec")
	}
	errs := testStatusStrategy.ValidateUpdate(ctx, &newService, &oldService)
	if len(errs) != 0 {
		t.Errorf("Unexpected error %v", errs)
	}
}

func makeServiceWithIPFamilies(ipfamilies []api.IPFamily, ipFamilyPolicy *api.IPFamilyPolicyType) *api.Service {
	return &api.Service{
		Spec: api.ServiceSpec{
			IPFamilies:     ipfamilies,
			IPFamilyPolicy: ipFamilyPolicy,
		},
	}
}

func TestDropDisabledField(t *testing.T) {
	requireDualStack := api.IPFamilyPolicyRequireDualStack
	preferDualStack := api.IPFamilyPolicyPreferDualStack
	singleStack := api.IPFamilyPolicySingleStack

	testCases := []struct {
		name            string
		enableDualStack bool
		svc             *api.Service
		oldSvc          *api.Service
		compareSvc      *api.Service
	}{
		{
			name:            "not dual stack, field not used",
			enableDualStack: false,
			svc:             makeServiceWithIPFamilies(nil, nil),
			oldSvc:          nil,
			compareSvc:      makeServiceWithIPFamilies(nil, nil),
		},
		{
			name:            "not dual stack, field used in old and new",
			enableDualStack: false,
			svc:             makeServiceWithIPFamilies([]api.IPFamily{api.IPv4Protocol}, nil),
			oldSvc:          makeServiceWithIPFamilies([]api.IPFamily{api.IPv4Protocol}, nil),
			compareSvc:      makeServiceWithIPFamilies([]api.IPFamily{api.IPv4Protocol}, nil),
		},
		{
			name:            "dualstack, field used",
			enableDualStack: true,
			svc:             makeServiceWithIPFamilies([]api.IPFamily{api.IPv6Protocol}, nil),
			oldSvc:          nil,
			compareSvc:      makeServiceWithIPFamilies([]api.IPFamily{api.IPv6Protocol}, nil),
		},
		/* preferDualStack field */
		{
			name:            "not dual stack, fields is not use",
			enableDualStack: false,
			svc:             makeServiceWithIPFamilies(nil, nil),
			oldSvc:          nil,
			compareSvc:      makeServiceWithIPFamilies(nil, nil),
		},
		{
			name:            "not dual stack, fields used in new, not in old",
			enableDualStack: false,
			svc:             makeServiceWithIPFamilies(nil, &preferDualStack),
			oldSvc:          nil,
			compareSvc:      makeServiceWithIPFamilies(nil, nil),
		},
		{
			name:            "not dual stack, fields used in new, not in old",
			enableDualStack: false,
			svc:             makeServiceWithIPFamilies(nil, &requireDualStack),
			oldSvc:          nil,
			compareSvc:      makeServiceWithIPFamilies(nil, nil),
		},

		{
			name:            "not dual stack, fields not used in old (single stack)",
			enableDualStack: false,
			svc:             makeServiceWithIPFamilies(nil, nil),
			oldSvc:          makeServiceWithIPFamilies(nil, &singleStack),
			compareSvc:      makeServiceWithIPFamilies(nil, nil),
		},
		{
			name:            "dualstack, field used",
			enableDualStack: true,
			svc:             makeServiceWithIPFamilies(nil, &singleStack),
			oldSvc:          nil,
			compareSvc:      makeServiceWithIPFamilies(nil, &singleStack),
		},

		/* add more tests for other dropped fields as needed */
	}
	for _, tc := range testCases {
		func() {
			defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.IPv6DualStack, tc.enableDualStack)()
			old := tc.oldSvc.DeepCopy()

			// to test against user using IPFamily not set on cluster
			svcStrategy := svcStrategy{ipFamilies: []api.IPFamily{api.IPv4Protocol}}
			svcStrategy.dropServiceDisabledFields(tc.svc, tc.oldSvc)

			// old node  should never be changed
			if !reflect.DeepEqual(tc.oldSvc, old) {
				t.Errorf("%v: old svc changed: %v", tc.name, diff.ObjectReflectDiff(tc.oldSvc, old))
			}

			if !reflect.DeepEqual(tc.svc, tc.compareSvc) {
				t.Errorf("%v: unexpected svc spec: %v", tc.name, diff.ObjectReflectDiff(tc.svc, tc.compareSvc))
			}
		}()
	}

}

func TestNormalizeClusterIPs(t *testing.T) {
	testCases := []struct {
		name               string
		oldService         *api.Service
		newService         *api.Service
		expectedClusterIP  string
		expectedClusterIPs []string
	}{

		{
			name:       "new - only clusterip used",
			oldService: nil,
			newService: &api.Service{
				Spec: api.ServiceSpec{
					ClusterIP:  "10.0.0.10",
					ClusterIPs: nil,
				},
			},
			expectedClusterIP:  "10.0.0.10",
			expectedClusterIPs: []string{"10.0.0.10"},
		},

		{
			name:       "new - only clusterips used",
			oldService: nil,
			newService: &api.Service{
				Spec: api.ServiceSpec{
					ClusterIP:  "",
					ClusterIPs: []string{"10.0.0.10"},
				},
			},
			expectedClusterIP:  "", // this is a validation issue, and validation will catch it
			expectedClusterIPs: []string{"10.0.0.10"},
		},

		{
			name:       "new - both used",
			oldService: nil,
			newService: &api.Service{
				Spec: api.ServiceSpec{
					ClusterIP:  "10.0.0.10",
					ClusterIPs: []string{"10.0.0.10"},
				},
			},
			expectedClusterIP:  "10.0.0.10",
			expectedClusterIPs: []string{"10.0.0.10"},
		},

		{
			name: "update - no change",
			oldService: &api.Service{
				Spec: api.ServiceSpec{
					ClusterIP:  "10.0.0.10",
					ClusterIPs: []string{"10.0.0.10"},
				},
			},
			newService: &api.Service{
				Spec: api.ServiceSpec{
					ClusterIP:  "10.0.0.10",
					ClusterIPs: []string{"10.0.0.10"},
				},
			},
			expectedClusterIP:  "10.0.0.10",
			expectedClusterIPs: []string{"10.0.0.10"},
		},

		{
			name: "update - malformed change",
			oldService: &api.Service{
				Spec: api.ServiceSpec{
					ClusterIP:  "10.0.0.10",
					ClusterIPs: []string{"10.0.0.10"},
				},
			},
			newService: &api.Service{
				Spec: api.ServiceSpec{
					ClusterIP:  "10.0.0.11",
					ClusterIPs: []string{"10.0.0.11"},
				},
			},
			expectedClusterIP:  "10.0.0.11",
			expectedClusterIPs: []string{"10.0.0.11"},
		},

		{
			name: "update - malformed change on secondary ip",
			oldService: &api.Service{
				Spec: api.ServiceSpec{
					ClusterIP:  "10.0.0.10",
					ClusterIPs: []string{"10.0.0.10", "2000::1"},
				},
			},
			newService: &api.Service{
				Spec: api.ServiceSpec{
					ClusterIP:  "10.0.0.11",
					ClusterIPs: []string{"10.0.0.11", "3000::1"},
				},
			},
			expectedClusterIP:  "10.0.0.11",
			expectedClusterIPs: []string{"10.0.0.11", "3000::1"},
		},

		{
			name: "update - upgrade",
			oldService: &api.Service{
				Spec: api.ServiceSpec{
					ClusterIP:  "10.0.0.10",
					ClusterIPs: []string{"10.0.0.10"},
				},
			},
			newService: &api.Service{
				Spec: api.ServiceSpec{
					ClusterIP:  "10.0.0.10",
					ClusterIPs: []string{"10.0.0.10", "2000::1"},
				},
			},
			expectedClusterIP:  "10.0.0.10",
			expectedClusterIPs: []string{"10.0.0.10", "2000::1"},
		},
		{
			name: "update - downgrade",
			oldService: &api.Service{
				Spec: api.ServiceSpec{
					ClusterIP:  "10.0.0.10",
					ClusterIPs: []string{"10.0.0.10", "2000::1"},
				},
			},
			newService: &api.Service{
				Spec: api.ServiceSpec{
					ClusterIP:  "10.0.0.10",
					ClusterIPs: []string{"10.0.0.10"},
				},
			},
			expectedClusterIP:  "10.0.0.10",
			expectedClusterIPs: []string{"10.0.0.10"},
		},

		{
			name: "update - user cleared cluster IP",
			oldService: &api.Service{
				Spec: api.ServiceSpec{
					ClusterIP:  "10.0.0.10",
					ClusterIPs: []string{"10.0.0.10"},
				},
			},
			newService: &api.Service{
				Spec: api.ServiceSpec{
					ClusterIP:  "",
					ClusterIPs: []string{"10.0.0.10"},
				},
			},
			expectedClusterIP:  "",
			expectedClusterIPs: nil,
		},

		{
			name: "update - user cleared clusterIPs", // *MUST* REMAIN FOR OLD CLIENTS
			oldService: &api.Service{
				Spec: api.ServiceSpec{
					ClusterIP:  "10.0.0.10",
					ClusterIPs: []string{"10.0.0.10"},
				},
			},
			newService: &api.Service{
				Spec: api.ServiceSpec{
					ClusterIP:  "10.0.0.10",
					ClusterIPs: nil,
				},
			},
			expectedClusterIP:  "10.0.0.10",
			expectedClusterIPs: []string{"10.0.0.10"},
		},

		{
			name: "update - user cleared both",
			oldService: &api.Service{
				Spec: api.ServiceSpec{
					ClusterIP:  "10.0.0.10",
					ClusterIPs: []string{"10.0.0.10"},
				},
			},
			newService: &api.Service{
				Spec: api.ServiceSpec{
					ClusterIP:  "",
					ClusterIPs: nil,
				},
			},
			expectedClusterIP:  "",
			expectedClusterIPs: nil,
		},

		{
			name: "update - user cleared ClusterIP but changed clusterIPs",
			oldService: &api.Service{
				Spec: api.ServiceSpec{
					ClusterIP:  "10.0.0.10",
					ClusterIPs: []string{"10.0.0.10"},
				},
			},
			newService: &api.Service{
				Spec: api.ServiceSpec{
					ClusterIP:  "",
					ClusterIPs: []string{"10.0.0.11"},
				},
			},
			expectedClusterIP:  "", /* validation catches this */
			expectedClusterIPs: []string{"10.0.0.11"},
		},

		{
			name: "update - user cleared ClusterIPs but changed ClusterIP",
			oldService: &api.Service{
				Spec: api.ServiceSpec{
					ClusterIP:  "10.0.0.10",
					ClusterIPs: []string{"10.0.0.10", "2000::1"},
				},
			},
			newService: &api.Service{
				Spec: api.ServiceSpec{
					ClusterIP:  "10.0.0.11",
					ClusterIPs: nil,
				},
			},
			expectedClusterIP:  "10.0.0.11",
			expectedClusterIPs: nil,
		},

		{
			name: "update - user changed from None to ClusterIP",
			oldService: &api.Service{
				Spec: api.ServiceSpec{
					ClusterIP:  "None",
					ClusterIPs: []string{"None"},
				},
			},
			newService: &api.Service{
				Spec: api.ServiceSpec{
					ClusterIP:  "10.0.0.10",
					ClusterIPs: []string{"None"},
				},
			},
			expectedClusterIP:  "10.0.0.10",
			expectedClusterIPs: []string{"10.0.0.10"},
		},

		{
			name: "update - user changed from ClusterIP to None",
			oldService: &api.Service{
				Spec: api.ServiceSpec{
					ClusterIP:  "10.0.0.10",
					ClusterIPs: []string{"10.0.0.10"},
				},
			},
			newService: &api.Service{
				Spec: api.ServiceSpec{
					ClusterIP:  "None",
					ClusterIPs: []string{"10.0.0.10"},
				},
			},
			expectedClusterIP:  "None",
			expectedClusterIPs: []string{"None"},
		},

		{
			name: "update - user changed from ClusterIP to None and changed ClusterIPs in a dual stack (new client making a mistake)",
			oldService: &api.Service{
				Spec: api.ServiceSpec{
					ClusterIP:  "10.0.0.10",
					ClusterIPs: []string{"10.0.0.10", "2000::1"},
				},
			},
			newService: &api.Service{
				Spec: api.ServiceSpec{
					ClusterIP:  "None",
					ClusterIPs: []string{"10.0.0.11", "2000::1"},
				},
			},
			expectedClusterIP:  "None",
			expectedClusterIPs: []string{"10.0.0.11", "2000::1"},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			normalizeClusterIPs(tc.oldService, tc.newService)

			if tc.newService == nil {
				t.Fatalf("unexpected new service to be nil")
			}

			if tc.newService.Spec.ClusterIP != tc.expectedClusterIP {
				t.Fatalf("expected clusterIP [%v] got [%v]", tc.expectedClusterIP, tc.newService.Spec.ClusterIP)
			}

			if len(tc.newService.Spec.ClusterIPs) != len(tc.expectedClusterIPs) {
				t.Fatalf("expected  clusterIPs %v got %v", tc.expectedClusterIPs, tc.newService.Spec.ClusterIPs)
			}

			for idx, clusterIP := range tc.newService.Spec.ClusterIPs {
				if clusterIP != tc.expectedClusterIPs[idx] {
					t.Fatalf("expected clusterIP [%v] at index[%v] got [%v]", tc.expectedClusterIPs[idx], idx, tc.newService.Spec.ClusterIPs[idx])

				}
			}
		})
	}

}

func TestClearClusterIPRelatedFields(t *testing.T) {
	//
	// NOTE the data fed to this test assums that ClusterIPs normalization is
	// already done check PrepareFor*(..) strategy
	//
	singleStack := api.IPFamilyPolicySingleStack
	requireDualStack := api.IPFamilyPolicyRequireDualStack
	testCases := []struct {
		name        string
		oldService  *api.Service
		newService  *api.Service
		shouldClear bool
	}{
		{
			name:        "should clear, single stack converting to external name",
			shouldClear: true,

			oldService: &api.Service{
				Spec: api.ServiceSpec{
					Type:           api.ServiceTypeClusterIP,
					IPFamilyPolicy: &singleStack,
					ClusterIP:      "10.0.0.4",
					ClusterIPs:     []string{"10.0.0.4"},
					IPFamilies:     []api.IPFamily{api.IPv4Protocol},
				},
			},
			newService: &api.Service{
				Spec: api.ServiceSpec{
					Type:           api.ServiceTypeExternalName,
					IPFamilyPolicy: &singleStack,
					ClusterIP:      "",
					ClusterIPs:     nil,
					IPFamilies:     []api.IPFamily{api.IPv4Protocol},
				},
			},
		},

		{
			name:        "should clear, dual stack converting to external name(normalization removed all ips)",
			shouldClear: true,

			oldService: &api.Service{
				Spec: api.ServiceSpec{
					Type:           api.ServiceTypeClusterIP,
					IPFamilyPolicy: &requireDualStack,
					ClusterIP:      "2000::1",
					ClusterIPs:     []string{"2000::1", "10.0.0.4"},
					IPFamilies:     []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol},
				},
			},
			newService: &api.Service{
				Spec: api.ServiceSpec{
					Type:           api.ServiceTypeExternalName,
					IPFamilyPolicy: &singleStack,
					ClusterIP:      "",
					ClusterIPs:     nil,
					IPFamilies:     []api.IPFamily{api.IPv4Protocol},
				},
			},
		},

		{
			name:        "should NOT clear, single stack converting to external name ClusterIPs was not cleared",
			shouldClear: false,

			oldService: &api.Service{
				Spec: api.ServiceSpec{
					Type:           api.ServiceTypeClusterIP,
					IPFamilyPolicy: &singleStack,
					ClusterIP:      "2000::1",
					ClusterIPs:     []string{"2000::1"},
					IPFamilies:     []api.IPFamily{api.IPv6Protocol},
				},
			},
			newService: &api.Service{
				Spec: api.ServiceSpec{
					Type:           api.ServiceTypeExternalName,
					IPFamilyPolicy: &singleStack,
					ClusterIP:      "2000::1",
					ClusterIPs:     []string{"2000::1"},
					IPFamilies:     []api.IPFamily{api.IPv4Protocol},
				},
			},
		},

		{
			name:        "should NOT clear, dualstack cleared primary and changed ClusterIPs",
			shouldClear: true,

			oldService: &api.Service{
				Spec: api.ServiceSpec{
					Type:           api.ServiceTypeClusterIP,
					IPFamilyPolicy: &requireDualStack,
					ClusterIP:      "2000::1",
					ClusterIPs:     []string{"2000::1", "10.0.0.4"},
					IPFamilies:     []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol},
				},
			},
			newService: &api.Service{
				Spec: api.ServiceSpec{
					Type:           api.ServiceTypeExternalName,
					IPFamilyPolicy: &singleStack,
					ClusterIP:      "",
					ClusterIPs:     []string{"2000::1", "10.0.0.5"},
					IPFamilies:     []api.IPFamily{api.IPv4Protocol},
				},
			},
		},
		{
			name:        "should clear, dualstack user removed ClusterIPs",
			shouldClear: true,

			oldService: &api.Service{
				Spec: api.ServiceSpec{
					Type:           api.ServiceTypeClusterIP,
					IPFamilyPolicy: &requireDualStack,
					ClusterIP:      "2000::1",
					ClusterIPs:     []string{"2000::1", "10.0.0.4"},
					IPFamilies:     []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol},
				},
			},

			newService: &api.Service{
				Spec: api.ServiceSpec{
					Type:           api.ServiceTypeExternalName,
					IPFamilyPolicy: &singleStack,
					ClusterIP:      "",
					ClusterIPs:     nil,
					IPFamilies:     []api.IPFamily{api.IPv4Protocol},
				},
			},
		},
		{
			name:        "should NOT clear, dualstack service changing selector",
			shouldClear: false,

			oldService: &api.Service{
				Spec: api.ServiceSpec{
					Type:           api.ServiceTypeClusterIP,
					Selector:       map[string]string{"foo": "bar"},
					IPFamilyPolicy: &requireDualStack,
					ClusterIP:      "2000::1",
					ClusterIPs:     []string{"2000::1", "10.0.0.4"},
					IPFamilies:     []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol},
				},
			},
			newService: &api.Service{
				Spec: api.ServiceSpec{
					Type:           api.ServiceTypeClusterIP,
					Selector:       map[string]string{"foo": "baz"},
					IPFamilyPolicy: &singleStack,
					ClusterIP:      "2000::1",
					ClusterIPs:     []string{"2000::1", "10.0.0.4"},
					IPFamilies:     []api.IPFamily{api.IPv4Protocol},
				},
			},
		},
	}

	for _, testCase := range testCases {
		t.Run(testCase.name, func(t *testing.T) {
			clearClusterIPRelatedFields(testCase.newService, testCase.oldService)

			if testCase.shouldClear && len(testCase.newService.Spec.ClusterIPs) != 0 {
				t.Fatalf("expected clusterIPs to be cleared")
			}

			if testCase.shouldClear && len(testCase.newService.Spec.IPFamilies) != 0 {
				t.Fatalf("expected ipfamilies to be cleared")
			}

			if testCase.shouldClear && testCase.newService.Spec.IPFamilyPolicy != nil {
				t.Fatalf("expected ipfamilypolicy to be cleared")
			}

			if !testCase.shouldClear && len(testCase.newService.Spec.ClusterIPs) == 0 {
				t.Fatalf("expected clusterIPs NOT to be cleared")
			}

			if !testCase.shouldClear && len(testCase.newService.Spec.IPFamilies) == 0 {
				t.Fatalf("expected ipfamilies NOT to be cleared")
			}

			if !testCase.shouldClear && testCase.newService.Spec.IPFamilyPolicy == nil {
				t.Fatalf("expected ipfamilypolicy NOT to be cleared")
			}

		})
	}
}

func TestTrimFieldsForDualStackDowngrade(t *testing.T) {
	singleStack := api.IPFamilyPolicySingleStack
	preferDualStack := api.IPFamilyPolicyPreferDualStack
	requireDualStack := api.IPFamilyPolicyRequireDualStack
	testCases := []struct {
		name          string
		oldPolicy     *api.IPFamilyPolicyType
		oldClusterIPs []string
		oldFamilies   []api.IPFamily

		newPolicy          *api.IPFamilyPolicyType
		expectedClusterIPs []string
		expectedIPFamilies []api.IPFamily
	}{

		{
			name:               "no change single to single",
			oldPolicy:          &singleStack,
			oldClusterIPs:      []string{"10.10.10.10"},
			oldFamilies:        []api.IPFamily{api.IPv4Protocol},
			newPolicy:          &singleStack,
			expectedClusterIPs: []string{"10.10.10.10"},
			expectedIPFamilies: []api.IPFamily{api.IPv4Protocol},
		},

		{
			name:               "dualstack to dualstack (preferred)",
			oldPolicy:          &preferDualStack,
			oldClusterIPs:      []string{"10.10.10.10", "2000::1"},
			oldFamilies:        []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
			newPolicy:          &preferDualStack,
			expectedClusterIPs: []string{"10.10.10.10", "2000::1"},
			expectedIPFamilies: []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
		},

		{
			name:               "dualstack to dualstack (required)",
			oldPolicy:          &requireDualStack,
			oldClusterIPs:      []string{"10.10.10.10", "2000::1"},
			oldFamilies:        []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
			newPolicy:          &preferDualStack,
			expectedClusterIPs: []string{"10.10.10.10", "2000::1"},
			expectedIPFamilies: []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
		},

		{
			name:               "dualstack (preferred) to single",
			oldPolicy:          &preferDualStack,
			oldClusterIPs:      []string{"10.10.10.10", "2000::1"},
			oldFamilies:        []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
			newPolicy:          &singleStack,
			expectedClusterIPs: []string{"10.10.10.10"},
			expectedIPFamilies: []api.IPFamily{api.IPv4Protocol},
		},

		{
			name:               "dualstack (require) to single",
			oldPolicy:          &requireDualStack,
			oldClusterIPs:      []string{"2000::1", "10.10.10.10"},
			oldFamilies:        []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol},
			newPolicy:          &singleStack,
			expectedClusterIPs: []string{"2000::1"},
			expectedIPFamilies: []api.IPFamily{api.IPv6Protocol},
		},
	}
	// only when gate is on
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.IPv6DualStack, true)()
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			oldService := &api.Service{
				Spec: api.ServiceSpec{
					IPFamilyPolicy: tc.oldPolicy,
					ClusterIPs:     tc.oldClusterIPs,
					IPFamilies:     tc.oldFamilies,
				},
			}

			newService := oldService.DeepCopy()
			newService.Spec.IPFamilyPolicy = tc.newPolicy

			trimFieldsForDualStackDowngrade(newService, oldService)

			if len(newService.Spec.ClusterIPs) != len(tc.expectedClusterIPs) {
				t.Fatalf("unexpected clusterIPs. expected %v and got %v", tc.expectedClusterIPs, newService.Spec.ClusterIPs)
			}

			// compare clusterIPS
			for i, expectedIP := range tc.expectedClusterIPs {
				if expectedIP != newService.Spec.ClusterIPs[i] {
					t.Fatalf("unexpected clusterIPs. expected %v and got %v", tc.expectedClusterIPs, newService.Spec.ClusterIPs)
				}
			}

			// families
			if len(newService.Spec.IPFamilies) != len(tc.expectedIPFamilies) {
				t.Fatalf("unexpected ipfamilies. expected %v and got %v", tc.expectedIPFamilies, newService.Spec.IPFamilies)
			}

			// compare clusterIPS
			for i, expectedIPFamily := range tc.expectedIPFamilies {
				if expectedIPFamily != newService.Spec.IPFamilies[i] {
					t.Fatalf("unexpected ipfamilies. expected %v and got %v", tc.expectedIPFamilies, newService.Spec.IPFamilies)
				}
			}

		})
	}
}
