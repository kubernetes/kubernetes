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

func makeValidService() *api.Service {
	preferDual := api.IPFamilyPolicyPreferDualStack

	return &api.Service{
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
			Ports: []api.ServicePort{
				makeValidServicePort("p", "TCP", 8675),
				makeValidServicePort("q", "TCP", 309),
			},
			ClusterIP:      "1.2.3.4",
			ClusterIPs:     []string{"1.2.3.4", "5:6:7::8"},
			IPFamilyPolicy: &preferDual,
			IPFamilies:     []api.IPFamily{"IPv4", "IPv6"},
		},
	}
}

func makeValidServicePort(name string, proto api.Protocol, port int32) api.ServicePort {
	return api.ServicePort{
		Name:       name,
		Protocol:   proto,
		Port:       port,
		TargetPort: intstr.FromInt(int(port)),
	}
}

func makeValidServiceCustom(tweaks ...func(svc *api.Service)) *api.Service {
	svc := makeValidService()
	for _, fn := range tweaks {
		fn(svc)
	}
	return svc
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
		tc.tweakSvc(oldSvc, newSvc)
		ctx := genericapirequest.NewDefaultContext()
		err := rest.BeforeUpdate(strategy, ctx, runtime.Object(oldSvc), runtime.Object(newSvc))
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
	testStatusStrategy.PrepareForUpdate(ctx, newService, oldService)
	if newService.Status.LoadBalancer.Ingress[0].IP != "127.0.0.2" {
		t.Errorf("Service status updates should allow change of status fields")
	}
	if newService.Spec.SessionAffinity != "None" {
		t.Errorf("PrepareForUpdate should have preserved old spec")
	}
	errs := testStatusStrategy.ValidateUpdate(ctx, newService, oldService)
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
			dropServiceDisabledFields(tc.svc, tc.oldSvc)

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
			NormalizeClusterIPs(tc.oldService, tc.newService)

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

func TestDropTypeDependentFields(t *testing.T) {
	// Tweaks used below.
	setTypeExternalName := func(svc *api.Service) {
		svc.Spec.Type = api.ServiceTypeExternalName
	}
	setTypeNodePort := func(svc *api.Service) {
		svc.Spec.Type = api.ServiceTypeNodePort
	}
	setTypeClusterIP := func(svc *api.Service) {
		svc.Spec.Type = api.ServiceTypeClusterIP
	}
	setTypeLoadBalancer := func(svc *api.Service) {
		svc.Spec.Type = api.ServiceTypeLoadBalancer
	}
	clearClusterIPs := func(svc *api.Service) {
		svc.Spec.ClusterIP = ""
		svc.Spec.ClusterIPs = nil
	}
	changeClusterIPs := func(svc *api.Service) {
		svc.Spec.ClusterIP += "0"
		svc.Spec.ClusterIPs[0] += "0"
	}
	setNodePorts := func(svc *api.Service) {
		for i := range svc.Spec.Ports {
			svc.Spec.Ports[i].NodePort = int32(30000 + i)
		}
	}
	changeNodePorts := func(svc *api.Service) {
		for i := range svc.Spec.Ports {
			svc.Spec.Ports[i].NodePort += 100
		}
	}
	clearIPFamilies := func(svc *api.Service) {
		svc.Spec.IPFamilies = nil
	}
	changeIPFamilies := func(svc *api.Service) {
		svc.Spec.IPFamilies[0] = svc.Spec.IPFamilies[1]
	}
	clearIPFamilyPolicy := func(svc *api.Service) {
		svc.Spec.IPFamilyPolicy = nil
	}
	changeIPFamilyPolicy := func(svc *api.Service) {
		single := api.IPFamilyPolicySingleStack
		svc.Spec.IPFamilyPolicy = &single
	}
	addPort := func(svc *api.Service) {
		svc.Spec.Ports = append(svc.Spec.Ports, makeValidServicePort("new", "TCP", 0))
	}
	delPort := func(svc *api.Service) {
		svc.Spec.Ports = svc.Spec.Ports[0 : len(svc.Spec.Ports)-1]
	}
	changePort := func(svc *api.Service) {
		svc.Spec.Ports[0].Port += 100
		svc.Spec.Ports[0].Protocol = "UDP"
	}
	setHCNodePort := func(svc *api.Service) {
		svc.Spec.ExternalTrafficPolicy = api.ServiceExternalTrafficPolicyTypeLocal
		svc.Spec.HealthCheckNodePort = int32(32000)
	}
	changeHCNodePort := func(svc *api.Service) {
		svc.Spec.HealthCheckNodePort += 100
	}
	patches := func(fns ...func(svc *api.Service)) func(svc *api.Service) {
		return func(svc *api.Service) {
			for _, fn := range fns {
				fn(svc)
			}
		}
	}

	testCases := []struct {
		name   string
		svc    *api.Service
		patch  func(svc *api.Service)
		expect *api.Service
	}{
		{ // clusterIP cases
			name:   "don't clear clusterIP et al",
			svc:    makeValidService(),
			patch:  nil,
			expect: makeValidService(),
		}, {
			name:   "clear clusterIP et al",
			svc:    makeValidService(),
			patch:  setTypeExternalName,
			expect: makeValidServiceCustom(setTypeExternalName, clearClusterIPs, clearIPFamilies, clearIPFamilyPolicy),
		}, {
			name:   "don't clear changed clusterIP",
			svc:    makeValidService(),
			patch:  patches(setTypeExternalName, changeClusterIPs),
			expect: makeValidServiceCustom(setTypeExternalName, changeClusterIPs, clearIPFamilies, clearIPFamilyPolicy),
		}, {
			name:   "don't clear changed ipFamilies",
			svc:    makeValidService(),
			patch:  patches(setTypeExternalName, changeIPFamilies),
			expect: makeValidServiceCustom(setTypeExternalName, clearClusterIPs, changeIPFamilies, clearIPFamilyPolicy),
		}, {
			name:   "don't clear changed ipFamilyPolicy",
			svc:    makeValidService(),
			patch:  patches(setTypeExternalName, changeIPFamilyPolicy),
			expect: makeValidServiceCustom(setTypeExternalName, clearClusterIPs, clearIPFamilies, changeIPFamilyPolicy),
		}, { // nodePort cases
			name:   "don't clear nodePorts for type=NodePort",
			svc:    makeValidServiceCustom(setTypeNodePort, setNodePorts),
			patch:  nil,
			expect: makeValidServiceCustom(setTypeNodePort, setNodePorts),
		}, {
			name:   "don't clear nodePorts for type=LoadBalancer",
			svc:    makeValidServiceCustom(setTypeLoadBalancer, setNodePorts),
			patch:  nil,
			expect: makeValidServiceCustom(setTypeLoadBalancer, setNodePorts),
		}, {
			name:   "clear nodePorts",
			svc:    makeValidServiceCustom(setTypeLoadBalancer, setNodePorts),
			patch:  setTypeClusterIP,
			expect: makeValidService(),
		}, {
			name:   "don't clear changed nodePorts",
			svc:    makeValidServiceCustom(setTypeLoadBalancer, setNodePorts),
			patch:  patches(setTypeClusterIP, changeNodePorts),
			expect: makeValidServiceCustom(setNodePorts, changeNodePorts),
		}, {
			name:   "clear nodePorts when adding a port",
			svc:    makeValidServiceCustom(setTypeLoadBalancer, setNodePorts),
			patch:  patches(setTypeClusterIP, addPort),
			expect: makeValidServiceCustom(addPort),
		}, {
			name:   "don't clear nodePorts when adding a port with NodePort",
			svc:    makeValidServiceCustom(setTypeLoadBalancer, setNodePorts),
			patch:  patches(setTypeClusterIP, addPort, setNodePorts),
			expect: makeValidServiceCustom(addPort, setNodePorts),
		}, {
			name:   "clear nodePorts when removing a port",
			svc:    makeValidServiceCustom(setTypeLoadBalancer, setNodePorts),
			patch:  patches(setTypeClusterIP, delPort),
			expect: makeValidServiceCustom(delPort),
		}, {
			name:   "clear nodePorts when changing a port",
			svc:    makeValidServiceCustom(setTypeLoadBalancer, setNodePorts),
			patch:  patches(setTypeClusterIP, changePort),
			expect: makeValidServiceCustom(changePort),
		}, { // healthCheckNodePort cases
			name:   "don't clear healthCheckNodePort for type=LoadBalancer",
			svc:    makeValidServiceCustom(setTypeLoadBalancer, setHCNodePort),
			patch:  nil,
			expect: makeValidServiceCustom(setTypeLoadBalancer, setHCNodePort),
		}, {
			name:   "clear healthCheckNodePort",
			svc:    makeValidServiceCustom(setTypeLoadBalancer, setHCNodePort),
			patch:  setTypeClusterIP,
			expect: makeValidService(),
		}, {
			name:   "don't clear changed healthCheckNodePort",
			svc:    makeValidServiceCustom(setTypeLoadBalancer, setHCNodePort),
			patch:  patches(setTypeClusterIP, changeHCNodePort),
			expect: makeValidServiceCustom(setHCNodePort, changeHCNodePort),
		}}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			result := tc.svc.DeepCopy()
			if tc.patch != nil {
				tc.patch(result)
			}
			dropTypeDependentFields(result, tc.svc)
			if result.Spec.ClusterIP != tc.expect.Spec.ClusterIP {
				t.Errorf("expected clusterIP %q, got %q", tc.expect.Spec.ClusterIP, result.Spec.ClusterIP)
			}
			if !reflect.DeepEqual(result.Spec.ClusterIPs, tc.expect.Spec.ClusterIPs) {
				t.Errorf("expected clusterIPs %q, got %q", tc.expect.Spec.ClusterIP, result.Spec.ClusterIP)
			}
			if !reflect.DeepEqual(result.Spec.IPFamilies, tc.expect.Spec.IPFamilies) {
				t.Errorf("expected ipFamilies %q, got %q", tc.expect.Spec.IPFamilies, result.Spec.IPFamilies)
			}
			if !reflect.DeepEqual(result.Spec.IPFamilyPolicy, tc.expect.Spec.IPFamilyPolicy) {
				t.Errorf("expected ipFamilyPolicy %q, got %q", getIPFamilyPolicy(tc.expect), getIPFamilyPolicy(result))
			}
			for i := range result.Spec.Ports {
				resultPort := result.Spec.Ports[i].NodePort
				expectPort := tc.expect.Spec.Ports[i].NodePort
				if resultPort != expectPort {
					t.Errorf("failed %q: expected Ports[%d].NodePort %d, got %d", tc.name, i, expectPort, resultPort)
				}
			}
			if result.Spec.HealthCheckNodePort != tc.expect.Spec.HealthCheckNodePort {
				t.Errorf("failed %q: expected healthCheckNodePort %d, got %d", tc.name, tc.expect.Spec.HealthCheckNodePort, result.Spec.HealthCheckNodePort)
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
