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

package controlplane

import (
	"reflect"
	"testing"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
	genericapiserver "k8s.io/apiserver/pkg/server"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/kubernetes/fake"
	core "k8s.io/client-go/testing"
	"k8s.io/kubernetes/pkg/controlplane/reconcilers"
	corerest "k8s.io/kubernetes/pkg/registry/core/rest"
	netutils "k8s.io/utils/net"
)

func TestCreateOrUpdateMasterService(t *testing.T) {
	singleStack := corev1.IPFamilyPolicySingleStack
	ns := metav1.NamespaceDefault
	om := func(name string) metav1.ObjectMeta {
		return metav1.ObjectMeta{Namespace: ns, Name: name}
	}

	createTests := []struct {
		testName     string
		serviceName  string
		servicePorts []corev1.ServicePort
		serviceType  corev1.ServiceType
		expectCreate *corev1.Service // nil means none expected
	}{
		{
			testName:    "service does not exist",
			serviceName: "foo",
			servicePorts: []corev1.ServicePort{
				{Name: "foo", Port: 8080, Protocol: "TCP", TargetPort: intstr.FromInt32(8080)},
			},
			serviceType: corev1.ServiceTypeClusterIP,
			expectCreate: &corev1.Service{
				ObjectMeta: om("foo"),
				Spec: corev1.ServiceSpec{
					Ports: []corev1.ServicePort{
						{Name: "foo", Port: 8080, Protocol: "TCP", TargetPort: intstr.FromInt32(8080)},
					},
					Selector:        nil,
					ClusterIP:       "1.2.3.4",
					IPFamilyPolicy:  &singleStack,
					SessionAffinity: corev1.ServiceAffinityNone,
					Type:            corev1.ServiceTypeClusterIP,
				},
			},
		},
	}
	for _, test := range createTests {
		master := Controller{}
		fakeClient := fake.NewSimpleClientset()
		master.client = fakeClient
		master.CreateOrUpdateMasterServiceIfNeeded(test.serviceName, netutils.ParseIPSloppy("1.2.3.4"), test.servicePorts, test.serviceType, false)
		creates := []core.CreateAction{}
		for _, action := range fakeClient.Actions() {
			if action.GetVerb() == "create" {
				creates = append(creates, action.(core.CreateAction))
			}
		}
		if test.expectCreate != nil {
			if len(creates) != 1 {
				t.Errorf("case %q: unexpected creations: %v", test.testName, creates)
			} else {
				obj := creates[0].GetObject()
				if e, a := test.expectCreate.Spec, obj.(*corev1.Service).Spec; !reflect.DeepEqual(e, a) {
					t.Errorf("case %q: expected create:\n%#v\ngot:\n%#v\n", test.testName, e, a)
				}
			}
		}
		if test.expectCreate == nil && len(creates) > 1 {
			t.Errorf("case %q: no create expected, yet saw: %v", test.testName, creates)
		}
	}

	reconcileTests := []struct {
		testName     string
		serviceName  string
		servicePorts []corev1.ServicePort
		serviceType  corev1.ServiceType
		service      *corev1.Service
		expectUpdate *corev1.Service // nil means none expected
	}{
		{
			testName:    "service definition wrong port",
			serviceName: "foo",
			servicePorts: []corev1.ServicePort{
				{Name: "foo", Port: 8080, Protocol: "TCP", TargetPort: intstr.FromInt32(8080)},
			},
			serviceType: corev1.ServiceTypeClusterIP,
			service: &corev1.Service{
				ObjectMeta: om("foo"),
				Spec: corev1.ServiceSpec{
					Ports: []corev1.ServicePort{
						{Name: "foo", Port: 8000, Protocol: "TCP", TargetPort: intstr.FromInt32(8080)},
					},
					Selector:        nil,
					ClusterIP:       "1.2.3.4",
					SessionAffinity: corev1.ServiceAffinityNone,
					Type:            corev1.ServiceTypeClusterIP,
				},
			},
			expectUpdate: &corev1.Service{
				ObjectMeta: om("foo"),
				Spec: corev1.ServiceSpec{
					Ports: []corev1.ServicePort{
						{Name: "foo", Port: 8080, Protocol: "TCP", TargetPort: intstr.FromInt32(8080)},
					},
					Selector:        nil,
					ClusterIP:       "1.2.3.4",
					SessionAffinity: corev1.ServiceAffinityNone,
					Type:            corev1.ServiceTypeClusterIP,
				},
			},
		},
		{
			testName:    "service definition missing port",
			serviceName: "foo",
			servicePorts: []corev1.ServicePort{
				{Name: "foo", Port: 8080, Protocol: "TCP", TargetPort: intstr.FromInt32(8080)},
				{Name: "baz", Port: 1000, Protocol: "TCP", TargetPort: intstr.FromInt32(1000)},
			},
			serviceType: corev1.ServiceTypeClusterIP,
			service: &corev1.Service{
				ObjectMeta: om("foo"),
				Spec: corev1.ServiceSpec{
					Ports: []corev1.ServicePort{
						{Name: "foo", Port: 8080, Protocol: "TCP", TargetPort: intstr.FromInt32(8080)},
					},
					Selector:        nil,
					ClusterIP:       "1.2.3.4",
					SessionAffinity: corev1.ServiceAffinityNone,
					Type:            corev1.ServiceTypeClusterIP,
				},
			},
			expectUpdate: &corev1.Service{
				ObjectMeta: om("foo"),
				Spec: corev1.ServiceSpec{
					Ports: []corev1.ServicePort{
						{Name: "foo", Port: 8080, Protocol: "TCP", TargetPort: intstr.FromInt32(8080)},
						{Name: "baz", Port: 1000, Protocol: "TCP", TargetPort: intstr.FromInt32(1000)},
					},
					Selector:        nil,
					ClusterIP:       "1.2.3.4",
					SessionAffinity: corev1.ServiceAffinityNone,
					Type:            corev1.ServiceTypeClusterIP,
				},
			},
		},
		{
			testName:    "service definition incorrect port",
			serviceName: "foo",
			servicePorts: []corev1.ServicePort{
				{Name: "foo", Port: 8080, Protocol: "TCP", TargetPort: intstr.FromInt32(8080)},
			},
			serviceType: corev1.ServiceTypeClusterIP,
			service: &corev1.Service{
				ObjectMeta: om("foo"),
				Spec: corev1.ServiceSpec{
					Ports: []corev1.ServicePort{
						{Name: "bar", Port: 1000, Protocol: "UDP", TargetPort: intstr.FromInt32(1000)},
					},
					Selector:        nil,
					ClusterIP:       "1.2.3.4",
					SessionAffinity: corev1.ServiceAffinityNone,
					Type:            corev1.ServiceTypeClusterIP,
				},
			},
			expectUpdate: &corev1.Service{
				ObjectMeta: om("foo"),
				Spec: corev1.ServiceSpec{
					Ports: []corev1.ServicePort{
						{Name: "foo", Port: 8080, Protocol: "TCP", TargetPort: intstr.FromInt32(8080)},
					},
					Selector:        nil,
					ClusterIP:       "1.2.3.4",
					SessionAffinity: corev1.ServiceAffinityNone,
					Type:            corev1.ServiceTypeClusterIP,
				},
			},
		},
		{
			testName:    "service definition incorrect port name",
			serviceName: "foo",
			servicePorts: []corev1.ServicePort{
				{Name: "foo", Port: 8080, Protocol: "TCP", TargetPort: intstr.FromInt32(8080)},
			},
			serviceType: corev1.ServiceTypeClusterIP,
			service: &corev1.Service{
				ObjectMeta: om("foo"),
				Spec: corev1.ServiceSpec{
					Ports: []corev1.ServicePort{
						{Name: "foo", Port: 1000, Protocol: "UDP", TargetPort: intstr.FromInt32(1000)},
					},
					Selector:        nil,
					ClusterIP:       "1.2.3.4",
					SessionAffinity: corev1.ServiceAffinityNone,
					Type:            corev1.ServiceTypeClusterIP,
				},
			},
			expectUpdate: &corev1.Service{
				ObjectMeta: om("foo"),
				Spec: corev1.ServiceSpec{
					Ports: []corev1.ServicePort{
						{Name: "foo", Port: 8080, Protocol: "TCP", TargetPort: intstr.FromInt32(8080)},
					},
					Selector:        nil,
					ClusterIP:       "1.2.3.4",
					SessionAffinity: corev1.ServiceAffinityNone,
					Type:            corev1.ServiceTypeClusterIP,
				},
			},
		},
		{
			testName:    "service definition incorrect target port",
			serviceName: "foo",
			servicePorts: []corev1.ServicePort{
				{Name: "foo", Port: 8080, Protocol: "TCP", TargetPort: intstr.FromInt32(8080)},
			},
			serviceType: corev1.ServiceTypeClusterIP,
			service: &corev1.Service{
				ObjectMeta: om("foo"),
				Spec: corev1.ServiceSpec{
					Ports: []corev1.ServicePort{
						{Name: "foo", Port: 8080, Protocol: "TCP", TargetPort: intstr.FromInt32(1000)},
					},
					Selector:        nil,
					ClusterIP:       "1.2.3.4",
					SessionAffinity: corev1.ServiceAffinityNone,
					Type:            corev1.ServiceTypeClusterIP,
				},
			},
			expectUpdate: &corev1.Service{
				ObjectMeta: om("foo"),
				Spec: corev1.ServiceSpec{
					Ports: []corev1.ServicePort{
						{Name: "foo", Port: 8080, Protocol: "TCP", TargetPort: intstr.FromInt32(8080)},
					},
					Selector:        nil,
					ClusterIP:       "1.2.3.4",
					SessionAffinity: corev1.ServiceAffinityNone,
					Type:            corev1.ServiceTypeClusterIP,
				},
			},
		},
		{
			testName:    "service definition incorrect protocol",
			serviceName: "foo",
			servicePorts: []corev1.ServicePort{
				{Name: "foo", Port: 8080, Protocol: "TCP", TargetPort: intstr.FromInt32(8080)},
			},
			serviceType: corev1.ServiceTypeClusterIP,
			service: &corev1.Service{
				ObjectMeta: om("foo"),
				Spec: corev1.ServiceSpec{
					Ports: []corev1.ServicePort{
						{Name: "foo", Port: 8080, Protocol: "UDP", TargetPort: intstr.FromInt32(8080)},
					},
					Selector:        nil,
					ClusterIP:       "1.2.3.4",
					SessionAffinity: corev1.ServiceAffinityNone,
					Type:            corev1.ServiceTypeClusterIP,
				},
			},
			expectUpdate: &corev1.Service{
				ObjectMeta: om("foo"),
				Spec: corev1.ServiceSpec{
					Ports: []corev1.ServicePort{
						{Name: "foo", Port: 8080, Protocol: "TCP", TargetPort: intstr.FromInt32(8080)},
					},
					Selector:        nil,
					ClusterIP:       "1.2.3.4",
					SessionAffinity: corev1.ServiceAffinityNone,
					Type:            corev1.ServiceTypeClusterIP,
				},
			},
		},
		{
			testName:    "service definition has incorrect type",
			serviceName: "foo",
			servicePorts: []corev1.ServicePort{
				{Name: "foo", Port: 8080, Protocol: "TCP", TargetPort: intstr.FromInt32(8080)},
			},
			serviceType: corev1.ServiceTypeClusterIP,
			service: &corev1.Service{
				ObjectMeta: om("foo"),
				Spec: corev1.ServiceSpec{
					Ports: []corev1.ServicePort{
						{Name: "foo", Port: 8080, Protocol: "TCP", TargetPort: intstr.FromInt32(8080)},
					},
					Selector:        nil,
					ClusterIP:       "1.2.3.4",
					SessionAffinity: corev1.ServiceAffinityNone,
					Type:            corev1.ServiceTypeNodePort,
				},
			},
			expectUpdate: &corev1.Service{
				ObjectMeta: om("foo"),
				Spec: corev1.ServiceSpec{
					Ports: []corev1.ServicePort{
						{Name: "foo", Port: 8080, Protocol: "TCP", TargetPort: intstr.FromInt32(8080)},
					},
					Selector:        nil,
					ClusterIP:       "1.2.3.4",
					SessionAffinity: corev1.ServiceAffinityNone,
					Type:            corev1.ServiceTypeClusterIP,
				},
			},
		},
		{
			testName:    "service definition satisfies",
			serviceName: "foo",
			servicePorts: []corev1.ServicePort{
				{Name: "foo", Port: 8080, Protocol: "TCP", TargetPort: intstr.FromInt32(8080)},
			},
			serviceType: corev1.ServiceTypeClusterIP,
			service: &corev1.Service{
				ObjectMeta: om("foo"),
				Spec: corev1.ServiceSpec{
					Ports: []corev1.ServicePort{
						{Name: "foo", Port: 8080, Protocol: "TCP", TargetPort: intstr.FromInt32(8080)},
					},
					Selector:        nil,
					ClusterIP:       "1.2.3.4",
					SessionAffinity: corev1.ServiceAffinityNone,
					Type:            corev1.ServiceTypeClusterIP,
				},
			},
			expectUpdate: nil,
		},
	}
	for _, test := range reconcileTests {
		master := Controller{}
		fakeClient := fake.NewSimpleClientset(test.service)
		master.client = fakeClient
		err := master.CreateOrUpdateMasterServiceIfNeeded(test.serviceName, netutils.ParseIPSloppy("1.2.3.4"), test.servicePorts, test.serviceType, true)
		if err != nil {
			t.Errorf("case %q: unexpected error: %v", test.testName, err)
		}
		updates := []core.UpdateAction{}
		for _, action := range fakeClient.Actions() {
			if action.GetVerb() == "update" {
				updates = append(updates, action.(core.UpdateAction))
			}
		}
		if test.expectUpdate != nil {
			if len(updates) != 1 {
				t.Errorf("case %q: unexpected updates: %v", test.testName, updates)
			} else {
				obj := updates[0].GetObject()
				if e, a := test.expectUpdate.Spec, obj.(*corev1.Service).Spec; !reflect.DeepEqual(e, a) {
					t.Errorf("case %q: expected update:\n%#v\ngot:\n%#v\n", test.testName, e, a)
				}
			}
		}
		if test.expectUpdate == nil && len(updates) > 0 {
			t.Errorf("case %q: no update expected, yet saw: %v", test.testName, updates)
		}
	}

	nonReconcileTests := []struct {
		testName     string
		serviceName  string
		servicePorts []corev1.ServicePort
		serviceType  corev1.ServiceType
		service      *corev1.Service
		expectUpdate *corev1.Service // nil means none expected
	}{
		{
			testName:    "service definition wrong port, no expected update",
			serviceName: "foo",
			servicePorts: []corev1.ServicePort{
				{Name: "foo", Port: 8080, Protocol: "TCP", TargetPort: intstr.FromInt32(8080)},
			},
			serviceType: corev1.ServiceTypeClusterIP,
			service: &corev1.Service{
				ObjectMeta: om("foo"),
				Spec: corev1.ServiceSpec{
					Ports: []corev1.ServicePort{
						{Name: "foo", Port: 1000, Protocol: "TCP", TargetPort: intstr.FromInt32(1000)},
					},
					Selector:        nil,
					ClusterIP:       "1.2.3.4",
					SessionAffinity: corev1.ServiceAffinityNone,
					Type:            corev1.ServiceTypeClusterIP,
				},
			},
			expectUpdate: nil,
		},
	}
	for _, test := range nonReconcileTests {
		master := Controller{}
		fakeClient := fake.NewSimpleClientset(test.service)
		master.client = fakeClient
		err := master.CreateOrUpdateMasterServiceIfNeeded(test.serviceName, netutils.ParseIPSloppy("1.2.3.4"), test.servicePorts, test.serviceType, false)
		if err != nil {
			t.Errorf("case %q: unexpected error: %v", test.testName, err)
		}
		updates := []core.UpdateAction{}
		for _, action := range fakeClient.Actions() {
			if action.GetVerb() == "update" {
				updates = append(updates, action.(core.UpdateAction))
			}
		}
		if test.expectUpdate != nil {
			if len(updates) != 1 {
				t.Errorf("case %q: unexpected updates: %v", test.testName, updates)
			} else {
				obj := updates[0].GetObject()
				if e, a := test.expectUpdate.Spec, obj.(*corev1.Service).Spec; !reflect.DeepEqual(e, a) {
					t.Errorf("case %q: expected update:\n%#v\ngot:\n%#v\n", test.testName, e, a)
				}
			}
		}
		if test.expectUpdate == nil && len(updates) > 0 {
			t.Errorf("case %q: no update expected, yet saw: %v", test.testName, updates)
		}
	}
}

func Test_completedConfig_NewBootstrapController(t *testing.T) {

	_, ipv4cidr, err := netutils.ParseCIDRSloppy("192.168.0.0/24")
	if err != nil {
		t.Fatalf("Unexpected error %v", err)
	}

	_, ipv6cidr, err := netutils.ParseCIDRSloppy("2001:db8::/112")
	if err != nil {
		t.Fatalf("Unexpected error %v", err)
	}

	ipv4address := netutils.ParseIPSloppy("192.168.1.1")
	ipv6address := netutils.ParseIPSloppy("2001:db8::1")

	type args struct {
		legacyRESTStorage corerest.LegacyRESTStorage
		client            kubernetes.Interface
	}
	tests := []struct {
		name        string
		config      genericapiserver.Config
		extraConfig *ExtraConfig
		args        args
		wantErr     bool
	}{
		{
			name: "master endpoint reconciler - IPv4 families",
			extraConfig: &ExtraConfig{
				EndpointReconcilerType: reconcilers.MasterCountReconcilerType,
				ServiceIPRange:         *ipv4cidr,
			},
			config: genericapiserver.Config{
				PublicAddress: ipv4address,
				SecureServing: &genericapiserver.SecureServingInfo{Listener: fakeLocalhost443Listener{}},
			},
			wantErr: false,
		},
		{
			name: "master endpoint reconciler - IPv6 families",
			extraConfig: &ExtraConfig{
				EndpointReconcilerType: reconcilers.MasterCountReconcilerType,
				ServiceIPRange:         *ipv6cidr,
			},
			config: genericapiserver.Config{
				PublicAddress: ipv6address,
				SecureServing: &genericapiserver.SecureServingInfo{Listener: fakeLocalhost443Listener{}},
			},
			wantErr: false,
		},
		{
			name: "master endpoint reconciler - wrong IP families",
			extraConfig: &ExtraConfig{
				EndpointReconcilerType: reconcilers.MasterCountReconcilerType,
				ServiceIPRange:         *ipv4cidr,
			},
			config: genericapiserver.Config{
				PublicAddress: ipv6address,
				SecureServing: &genericapiserver.SecureServingInfo{Listener: fakeLocalhost443Listener{}},
			},
			wantErr: true,
		},
		{
			name: "master endpoint reconciler - wrong IP families",
			extraConfig: &ExtraConfig{
				EndpointReconcilerType: reconcilers.MasterCountReconcilerType,
				ServiceIPRange:         *ipv6cidr,
			},
			config: genericapiserver.Config{
				PublicAddress: ipv4address,
				SecureServing: &genericapiserver.SecureServingInfo{Listener: fakeLocalhost443Listener{}},
			},
			wantErr: true,
		},
		{
			name: "lease endpoint reconciler - IPv4 families",
			extraConfig: &ExtraConfig{
				EndpointReconcilerType: reconcilers.LeaseEndpointReconcilerType,
				ServiceIPRange:         *ipv4cidr,
			},
			config: genericapiserver.Config{
				PublicAddress: ipv4address,
				SecureServing: &genericapiserver.SecureServingInfo{Listener: fakeLocalhost443Listener{}},
			},
			wantErr: false,
		},
		{
			name: "lease endpoint reconciler - IPv6 families",
			extraConfig: &ExtraConfig{
				EndpointReconcilerType: reconcilers.LeaseEndpointReconcilerType,
				ServiceIPRange:         *ipv6cidr,
			},
			config: genericapiserver.Config{
				PublicAddress: ipv6address,
				SecureServing: &genericapiserver.SecureServingInfo{Listener: fakeLocalhost443Listener{}},
			},
			wantErr: false,
		},
		{
			name: "lease endpoint reconciler - wrong IP families",
			extraConfig: &ExtraConfig{
				EndpointReconcilerType: reconcilers.LeaseEndpointReconcilerType,
				ServiceIPRange:         *ipv4cidr,
			},
			config: genericapiserver.Config{
				PublicAddress: ipv6address,
				SecureServing: &genericapiserver.SecureServingInfo{Listener: fakeLocalhost443Listener{}},
			},
			wantErr: true,
		},
		{
			name: "lease endpoint reconciler - wrong IP families",
			extraConfig: &ExtraConfig{
				EndpointReconcilerType: reconcilers.LeaseEndpointReconcilerType,
				ServiceIPRange:         *ipv6cidr,
			},
			config: genericapiserver.Config{
				PublicAddress: ipv4address,
				SecureServing: &genericapiserver.SecureServingInfo{Listener: fakeLocalhost443Listener{}},
			},
			wantErr: true,
		},
		{
			name: "none endpoint reconciler - wrong IP families",
			extraConfig: &ExtraConfig{
				EndpointReconcilerType: reconcilers.NoneEndpointReconcilerType,
				ServiceIPRange:         *ipv4cidr,
			},
			config: genericapiserver.Config{
				PublicAddress: ipv6address,
				SecureServing: &genericapiserver.SecureServingInfo{Listener: fakeLocalhost443Listener{}},
			},
			wantErr: false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c := &completedConfig{
				GenericConfig: tt.config.Complete(nil),
				ExtraConfig:   tt.extraConfig,
			}
			_, err := c.NewBootstrapController(tt.args.legacyRESTStorage, tt.args.client)
			if (err != nil) != tt.wantErr {
				t.Errorf("completedConfig.NewBootstrapController() error = %v, wantErr %v", err, tt.wantErr)
				return
			}

		})
	}
}
