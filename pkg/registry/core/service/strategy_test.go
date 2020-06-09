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
					ClusterIP: "10.0.0.1",
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
					ClusterIP: "",
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
	defaultServiceIPFamily := api.IPv4Protocol
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
			IPFamily:        &defaultServiceIPFamily,
		},
	}
}

// TODO: This should be done on types that are not part of our API
func TestBeforeCreate(t *testing.T) {
	withIP := func(family *api.IPFamily, ip string) *api.Service {
		svc := makeValidService()
		svc.Spec.IPFamily = family
		svc.Spec.ClusterIP = ip
		return &svc
	}

	ipv4 := api.IPv4Protocol
	ipv6 := api.IPv6Protocol
	testCases := []struct {
		name               string
		cidr               string
		configureDualStack bool
		enableDualStack    bool
		in                 *api.Service
		expect             *api.Service
		expectErr          bool
	}{
		{
			name:   "does not set ipfamily when dual stack gate is disabled",
			cidr:   "10.0.0.0/16",
			in:     withIP(nil, ""),
			expect: withIP(nil, ""),
		},

		{
			name:   "clears ipfamily when dual stack gate is disabled",
			cidr:   "10.0.0.0/16",
			in:     withIP(&ipv4, ""),
			expect: withIP(nil, ""),
		},

		{
			name:            "allows ipfamily to configured ipv4 value",
			cidr:            "10.0.0.0/16",
			enableDualStack: true,
			in:              withIP(nil, ""),
			expect:          withIP(&ipv4, ""),
		},
		{
			name:               "allows ipfamily to configured ipv4 value when dual stack is in use",
			cidr:               "10.0.0.0/16",
			enableDualStack:    true,
			configureDualStack: true,
			in:                 withIP(nil, ""),
			expect:             withIP(&ipv4, ""),
		},
		{
			name:            "allows ipfamily to configured ipv6 value",
			cidr:            "fd00::/64",
			enableDualStack: true,
			in:              withIP(nil, ""),
			expect:          withIP(&ipv6, ""),
		},
		{
			name:               "allows ipfamily to configured ipv6 value when dual stack is in use",
			cidr:               "fd00::/64",
			enableDualStack:    true,
			configureDualStack: true,
			in:                 withIP(nil, ""),
			expect:             withIP(&ipv6, ""),
		},

		{
			name:            "rejects ipv6 ipfamily when single-stack ipv4",
			enableDualStack: true,
			cidr:            "10.0.0.0/16",
			in:              withIP(&ipv6, ""),
			expectErr:       true,
		},
		{
			name:            "rejects ipv4 ipfamily when single-stack ipv6",
			enableDualStack: true,
			cidr:            "fd00::/64",
			in:              withIP(&ipv4, ""),
			expectErr:       true,
		},
		{
			name:            "rejects implicit ipv4 ipfamily when single-stack ipv6",
			enableDualStack: true,
			cidr:            "fd00::/64",
			in:              withIP(nil, "10.0.1.0"),
			expectErr:       true,
		},
		{
			name:            "rejects implicit ipv6 ipfamily when single-stack ipv4",
			enableDualStack: true,
			cidr:            "10.0.0.0/16",
			in:              withIP(nil, "fd00::1"),
			expectErr:       true,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.IPv6DualStack, tc.enableDualStack)()
			testStrategy, _ := newStrategy(tc.cidr, tc.configureDualStack)
			ctx := genericapirequest.NewDefaultContext()
			err := rest.BeforeCreate(testStrategy, ctx, runtime.Object(tc.in))
			if tc.expectErr != (err != nil) {
				t.Fatalf("unexpected error: %v", err)
			}
			if err != nil {
				return
			}
			if tc.expect != nil && tc.in != nil {
				tc.expect.ObjectMeta = tc.in.ObjectMeta
			}
			if !reflect.DeepEqual(tc.expect, tc.in) {
				t.Fatalf("unexpected change: %s", diff.ObjectReflectDiff(tc.expect, tc.in))
			}
		})
	}
}

func TestBeforeUpdate(t *testing.T) {
	testCases := []struct {
		name            string
		enableDualStack bool
		defaultIPv6     bool
		allowSecondary  bool
		tweakSvc        func(oldSvc, newSvc *api.Service) // given basic valid services, each test case can customize them
		expectErr       bool
		expectObj       func(t *testing.T, svc *api.Service)
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
				oldSvc.Spec.ClusterIP = "1.2.3.4"
				newSvc.Spec.ClusterIP = "4.3.2.1"
			},
			expectErr: true,
		},
		{
			name:            "clear IP family is allowed (defaulted back by before update)",
			enableDualStack: true,
			tweakSvc: func(oldSvc, newSvc *api.Service) {
				oldSvc.Spec.IPFamily = nil
			},
			expectErr: false,
			expectObj: func(t *testing.T, svc *api.Service) {
				if svc.Spec.IPFamily == nil {
					t.Errorf("ipfamily was not defaulted")
				}
			},
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
		t.Run(tc.name, func(t *testing.T) {
			defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.IPv6DualStack, tc.enableDualStack)()
			var cidr string
			if tc.defaultIPv6 {
				cidr = "ffd0::/64"
			} else {
				cidr = "172.30.0.0/16"
			}
			strategy, _ := newStrategy(cidr, tc.allowSecondary)
			oldSvc := makeValidService()
			newSvc := makeValidService()
			tc.tweakSvc(&oldSvc, &newSvc)
			ctx := genericapirequest.NewDefaultContext()
			err := rest.BeforeUpdate(strategy, ctx, runtime.Object(&newSvc), runtime.Object(&oldSvc))
			if tc.expectObj != nil {
				tc.expectObj(t, &newSvc)
			}
			if tc.expectErr && err == nil {
				t.Fatalf("unexpected non-error: %v", err)
			}
			if !tc.expectErr && err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
		})
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

func makeServiceWithIPFamily(ipFamily *api.IPFamily) *api.Service {
	return &api.Service{
		Spec: api.ServiceSpec{
			IPFamily: ipFamily,
		},
	}
}
func TestDropDisabledField(t *testing.T) {
	ipv4Service := api.IPv4Protocol
	ipv6Service := api.IPv6Protocol
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
			svc:             makeServiceWithIPFamily(nil),
			oldSvc:          nil,
			compareSvc:      makeServiceWithIPFamily(nil),
		},
		{
			name:            "not dual stack, field used in new, not in old",
			enableDualStack: false,
			svc:             makeServiceWithIPFamily(&ipv4Service),
			oldSvc:          nil,
			compareSvc:      makeServiceWithIPFamily(nil),
		},
		{
			name:            "not dual stack, field used in old and new",
			enableDualStack: false,
			svc:             makeServiceWithIPFamily(&ipv4Service),
			oldSvc:          makeServiceWithIPFamily(&ipv4Service),
			compareSvc:      makeServiceWithIPFamily(&ipv4Service),
		},
		{
			name:            "dualstack, field used",
			enableDualStack: true,
			svc:             makeServiceWithIPFamily(&ipv6Service),
			oldSvc:          nil,
			compareSvc:      makeServiceWithIPFamily(&ipv6Service),
		},
		{
			name:            "dualstack, field used, changed",
			enableDualStack: true,
			svc:             makeServiceWithIPFamily(&ipv6Service),
			oldSvc:          makeServiceWithIPFamily(&ipv4Service),
			compareSvc:      makeServiceWithIPFamily(&ipv6Service),
		},
		{
			name:            "dualstack, field used, not changed",
			enableDualStack: true,
			svc:             makeServiceWithIPFamily(&ipv6Service),
			oldSvc:          makeServiceWithIPFamily(&ipv6Service),
			compareSvc:      makeServiceWithIPFamily(&ipv6Service),
		},

		/* add more tests for other dropped fields as needed */
	}
	for _, tc := range testCases {
		func() {
			defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.IPv6DualStack, tc.enableDualStack)()
			old := tc.oldSvc.DeepCopy()
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
