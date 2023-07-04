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

package defaultservice

import (
	"net"
	"reflect"
	"testing"
	"time"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/util/intstr"
	v1informers "k8s.io/client-go/informers/core/v1"
	"k8s.io/client-go/kubernetes/fake"
	v1listers "k8s.io/client-go/listers/core/v1"
	core "k8s.io/client-go/testing"
	"k8s.io/client-go/tools/cache"
	"k8s.io/kubernetes/pkg/controlplane/reconcilers"
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
		t.Run(test.testName, func(t *testing.T) {
			master := Controller{}
			fakeClient := fake.NewSimpleClientset()
			serviceStore := cache.NewIndexer(cache.MetaNamespaceKeyFunc, cache.Indexers{})
			master.serviceLister = v1listers.NewServiceLister(serviceStore)
			master.client = fakeClient
			master.createOrUpdateMasterServiceIfNeeded(test.serviceName, netutils.ParseIPSloppy("1.2.3.4"), test.servicePorts, test.serviceType, false)
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
		})
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
		t.Run(test.testName, func(t *testing.T) {
			master := Controller{}
			fakeClient := fake.NewSimpleClientset(test.service)
			serviceInformer := v1informers.NewFilteredServiceInformer(fakeClient, metav1.NamespaceDefault, 12*time.Hour,
				cache.Indexers{},
				func(options *metav1.ListOptions) {
					options.FieldSelector = fields.OneTermEqualSelector("metadata.name", test.serviceName).String()
				})
			master.serviceInformer = serviceInformer
			serviceStore := serviceInformer.GetIndexer()
			err := serviceStore.Add(test.service)
			if err != nil {
				t.Fatalf("unexpected error adding service %v to the store: %v", test.service, err)
			}
			master.serviceLister = v1listers.NewServiceLister(serviceStore)
			master.client = fakeClient
			err = master.createOrUpdateMasterServiceIfNeeded(test.serviceName, netutils.ParseIPSloppy("1.2.3.4"), test.servicePorts, test.serviceType, true)
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
		})
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
		t.Run(test.testName, func(t *testing.T) {
			master := Controller{}
			fakeClient := fake.NewSimpleClientset(test.service)
			serviceInformer := v1informers.NewFilteredServiceInformer(fakeClient, metav1.NamespaceDefault, 12*time.Hour,
				cache.Indexers{},
				func(options *metav1.ListOptions) {
					options.FieldSelector = fields.OneTermEqualSelector("metadata.name", test.serviceName).String()
				})
			master.serviceInformer = serviceInformer
			serviceStore := serviceInformer.GetIndexer()
			err := serviceStore.Add(test.service)
			if err != nil {
				t.Fatalf("unexpected error adding service %v to the store: %v", test.service, err)
			}
			master.serviceLister = v1listers.NewServiceLister(serviceStore)

			err = master.createOrUpdateMasterServiceIfNeeded(test.serviceName, netutils.ParseIPSloppy("1.2.3.4"), test.servicePorts, test.serviceType, false)
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
		})
	}
}

// verify that if the endpoint reconciler is set
// the Service and Endpoints use the same IP family
func Test_NewController(t *testing.T) {

	ipv4address := netutils.ParseIPSloppy("192.168.1.1")
	ipv6address := netutils.ParseIPSloppy("2001:db8::1")

	tests := []struct {
		name       string
		reconciler reconcilers.Type
		serviceIP  net.IP
		endpointIP net.IP
		wantErr    bool
	}{
		{
			name:       "master endpoint reconciler - IPv4 families",
			reconciler: reconcilers.MasterCountReconcilerType,
			serviceIP:  ipv4address,
			endpointIP: ipv4address,
			wantErr:    false,
		},
		{
			name:       "master endpoint reconciler - IPv6 families",
			reconciler: reconcilers.MasterCountReconcilerType,
			serviceIP:  ipv6address,
			endpointIP: ipv6address,
			wantErr:    false,
		},
		{
			name:       "master endpoint reconciler - wrong IP families",
			reconciler: reconcilers.MasterCountReconcilerType,
			serviceIP:  ipv6address,
			endpointIP: ipv4address,
			wantErr:    true,
		},
		{
			name:       "master endpoint reconciler - wrong IP families",
			reconciler: reconcilers.MasterCountReconcilerType,
			serviceIP:  ipv4address,
			endpointIP: ipv6address,
			wantErr:    true,
		},
		{
			name:       "lease endpoint reconciler - IPv4 families",
			reconciler: reconcilers.LeaseEndpointReconcilerType,
			serviceIP:  ipv4address,
			endpointIP: ipv4address,
			wantErr:    false,
		},
		{
			name:       "lease endpoint reconciler - IPv6 families",
			reconciler: reconcilers.LeaseEndpointReconcilerType,
			serviceIP:  ipv6address,
			endpointIP: ipv6address,
			wantErr:    false,
		},
		{
			name:       "lease endpoint reconciler - wrong IP families",
			reconciler: reconcilers.LeaseEndpointReconcilerType,
			serviceIP:  ipv6address,
			endpointIP: ipv4address,
			wantErr:    true,
		},
		{
			name:       "lease endpoint reconciler - wrong IP families",
			reconciler: reconcilers.LeaseEndpointReconcilerType,
			serviceIP:  ipv4address,
			endpointIP: ipv6address,
			wantErr:    true,
		},
		{
			name:       "none endpoint reconciler - wrong IP families",
			reconciler: reconcilers.NoneEndpointReconcilerType,
			serviceIP:  ipv4address,
			endpointIP: ipv6address,
			wantErr:    false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := NewController(tt.serviceIP, 443, 30123, tt.endpointIP, 443, reconcilers.NewNoneEndpointReconciler(), 0, tt.reconciler, fake.NewSimpleClientset())
			if (err != nil) != tt.wantErr {
				t.Errorf("completedConfig.NewController() error = %v, wantErr %v", err, tt.wantErr)
				return
			}

		})
	}
}
