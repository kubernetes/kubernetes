/*
Copyright 2022 The Kubernetes Authors.

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

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/client-go/kubernetes/fake"
	k8stesting "k8s.io/client-go/testing"
	"k8s.io/kubernetes/pkg/controlplane/reconcilers"
	netutils "k8s.io/utils/net"
)

func TestCreateOrUpdateMasterServiceCreate(t *testing.T) {
	singleStack := v1.IPFamilyPolicySingleStack
	ns := metav1.NamespaceDefault
	om := func(name string) metav1.ObjectMeta {
		return metav1.ObjectMeta{
			Namespace: ns,
			Name:      name,
			Labels:    map[string]string{"component": "apiserver", "provider": "kubernetes"},
		}
	}

	type args struct {
		serviceIP         net.IP
		publicIP          net.IP
		servicePort       int
		serviceNodePort   int
		publicServicePort int
	}

	createTests := []struct {
		testName     string
		args         args
		expectCreate *v1.Service // nil means none expected
	}{
		{
			testName: "service does not exist",
			args: args{
				serviceIP:         netutils.ParseIPSloppy("1.2.3.4"),
				publicIP:          netutils.ParseIPSloppy("192.168.0.1"),
				servicePort:       443,
				serviceNodePort:   0,
				publicServicePort: 6443,
			},
			expectCreate: &v1.Service{
				ObjectMeta: om(serviceName),
				Spec: v1.ServiceSpec{
					Ports: []v1.ServicePort{
						{Name: "https", Port: 443, Protocol: "TCP", TargetPort: intstr.FromInt(6443)},
					},
					Selector:        nil,
					ClusterIP:       "1.2.3.4",
					IPFamilyPolicy:  &singleStack,
					SessionAffinity: v1.ServiceAffinityNone,
					Type:            v1.ServiceTypeClusterIP,
				},
			},
		},
	}
	for _, test := range createTests {
		t.Run(test.testName, func(t *testing.T) {
			fakeClient := fake.NewSimpleClientset()

			controller, err := NewController(
				test.args.serviceIP,
				test.args.servicePort,
				test.args.serviceNodePort,
				test.args.publicIP,
				test.args.publicServicePort,
				reconcilers.NewNoneEndpointReconciler(),
				0*time.Second,
				reconcilers.NoneEndpointReconcilerType,
				fakeClient)
			if err != nil {
				t.Fatal(err)
			}
			fakeClient.PrependReactor("create", "services", func(action k8stesting.Action) (bool, runtime.Object, error) {
				create := action.(k8stesting.CreateAction)
				controller.serviceInformer.GetIndexer().Add(create.GetObject())
				return true, create.GetObject(), nil
			})
			// create initial service
			err = controller.createOrUpdateMasterServiceIfNeeded()
			if err != nil {
				t.Errorf("case %q: unexpected error: %v", test.testName, err)
			}
			creates := []k8stesting.CreateAction{}
			for _, action := range fakeClient.Actions() {
				if action.GetVerb() == "create" {
					creates = append(creates, action.(k8stesting.CreateAction))
				}
			}
			if test.expectCreate != nil {
				if len(creates) != 1 {
					t.Errorf("case %q: unexpected creations: %v", test.testName, creates)
				} else {
					obj := creates[0].GetObject()
					if e, a := test.expectCreate.ObjectMeta, obj.(*v1.Service).ObjectMeta; !reflect.DeepEqual(e, a) {
						t.Errorf("case %q: expected create:\n%#v\ngot:\n%#v\n", test.testName, e, a)
					}
					if e, a := test.expectCreate.Spec, obj.(*v1.Service).Spec; !reflect.DeepEqual(e, a) {
						t.Errorf("case %q: expected create:\n%#v\ngot:\n%#v\n", test.testName, e, a)
					}
				}
			}
			if test.expectCreate == nil && len(creates) > 1 {
				t.Errorf("case %q: no create expected, yet saw: %v", test.testName, creates)
			}
		})
	}
}

func TestCreateOrUpdateMasterServiceReconcile(t *testing.T) {
	ns := metav1.NamespaceDefault
	singleStack := v1.IPFamilyPolicySingleStack
	om := func(name string) metav1.ObjectMeta {
		return metav1.ObjectMeta{
			Namespace: ns,
			Name:      name,
			Labels:    map[string]string{"component": "apiserver", "provider": "kubernetes"},
		}
	}

	reconcileTests := []struct {
		testName     string
		service      *v1.Service
		expectUpdate *v1.Service // nil means none expected
	}{
		{
			testName: "service definition wrong port",
			service: &v1.Service{
				ObjectMeta: om(serviceName),
				Spec: v1.ServiceSpec{
					Ports: []v1.ServicePort{
						{Name: "https", Port: 80, Protocol: "TCP", TargetPort: intstr.FromInt(6443)},
					},
					Selector:        nil,
					ClusterIP:       "1.2.3.4",
					IPFamilyPolicy:  &singleStack,
					SessionAffinity: v1.ServiceAffinityNone,
					Type:            v1.ServiceTypeClusterIP,
				},
			},
			expectUpdate: &v1.Service{
				ObjectMeta: om(serviceName),
				Spec: v1.ServiceSpec{
					Ports: []v1.ServicePort{
						{Name: "https", Port: 443, Protocol: "TCP", TargetPort: intstr.FromInt(6443)},
					},
					Selector:        nil,
					ClusterIP:       "1.2.3.4",
					IPFamilyPolicy:  &singleStack,
					SessionAffinity: v1.ServiceAffinityNone,
					Type:            v1.ServiceTypeClusterIP,
				},
			},
		},
		{
			testName: "service definition additional port",
			service: &v1.Service{
				ObjectMeta: om(serviceName),
				Spec: v1.ServiceSpec{
					Ports: []v1.ServicePort{
						{Name: "https", Port: 443, Protocol: "TCP", TargetPort: intstr.FromInt(6443)},
						{Name: "http", Port: 80, Protocol: "TCP", TargetPort: intstr.FromInt(8080)},
					},
					Selector:        nil,
					ClusterIP:       "1.2.3.4",
					IPFamilyPolicy:  &singleStack,
					SessionAffinity: v1.ServiceAffinityNone,
					Type:            v1.ServiceTypeClusterIP,
				},
			},
			expectUpdate: &v1.Service{
				ObjectMeta: om(serviceName),
				Spec: v1.ServiceSpec{
					Ports: []v1.ServicePort{
						{Name: "https", Port: 443, Protocol: "TCP", TargetPort: intstr.FromInt(6443)},
					},
					Selector:        nil,
					ClusterIP:       "1.2.3.4",
					IPFamilyPolicy:  &singleStack,
					SessionAffinity: v1.ServiceAffinityNone,
					Type:            v1.ServiceTypeClusterIP,
				},
			},
		},
		{
			testName: "service different protocol",
			service: &v1.Service{
				ObjectMeta: om(serviceName),
				Spec: v1.ServiceSpec{
					Ports: []v1.ServicePort{
						{Name: "https", Port: 443, Protocol: "UDP", TargetPort: intstr.FromInt(6443)},
					},
					Selector:        nil,
					ClusterIP:       "1.2.3.4",
					IPFamilyPolicy:  &singleStack,
					SessionAffinity: v1.ServiceAffinityNone,
					Type:            v1.ServiceTypeClusterIP,
				},
			},
			expectUpdate: &v1.Service{
				ObjectMeta: om(serviceName),
				Spec: v1.ServiceSpec{
					Ports: []v1.ServicePort{
						{Name: "https", Port: 443, Protocol: "TCP", TargetPort: intstr.FromInt(6443)},
					},
					Selector:        nil,
					ClusterIP:       "1.2.3.4",
					IPFamilyPolicy:  &singleStack,
					SessionAffinity: v1.ServiceAffinityNone,
					Type:            v1.ServiceTypeClusterIP,
				},
			},
		},
		{
			testName: "service definition incorrect port name",
			service: &v1.Service{
				ObjectMeta: om(serviceName),
				Spec: v1.ServiceSpec{
					Ports: []v1.ServicePort{
						{Name: "foo", Port: 443, Protocol: "TCP", TargetPort: intstr.FromInt(6443)},
					},
					Selector:        nil,
					ClusterIP:       "1.2.3.4",
					IPFamilyPolicy:  &singleStack,
					SessionAffinity: v1.ServiceAffinityNone,
					Type:            v1.ServiceTypeClusterIP,
				},
			},
			expectUpdate: &v1.Service{
				ObjectMeta: om(serviceName),
				Spec: v1.ServiceSpec{
					Ports: []v1.ServicePort{
						{Name: "https", Port: 443, Protocol: "TCP", TargetPort: intstr.FromInt(6443)},
					},
					Selector:        nil,
					ClusterIP:       "1.2.3.4",
					IPFamilyPolicy:  &singleStack,
					SessionAffinity: v1.ServiceAffinityNone,
					Type:            v1.ServiceTypeClusterIP,
				},
			},
		},
		{
			testName: "service definition incorrect target port",
			service: &v1.Service{
				ObjectMeta: om(serviceName),
				Spec: v1.ServiceSpec{
					Ports: []v1.ServicePort{
						{Name: "https", Port: 443, Protocol: "TCP", TargetPort: intstr.FromInt(9443)},
					},
					Selector:        nil,
					ClusterIP:       "1.2.3.4",
					IPFamilyPolicy:  &singleStack,
					SessionAffinity: v1.ServiceAffinityNone,
					Type:            v1.ServiceTypeClusterIP,
				},
			},
			expectUpdate: &v1.Service{
				ObjectMeta: om(serviceName),
				Spec: v1.ServiceSpec{
					Ports: []v1.ServicePort{
						{Name: "https", Port: 443, Protocol: "TCP", TargetPort: intstr.FromInt(6443)},
					},
					Selector:        nil,
					ClusterIP:       "1.2.3.4",
					IPFamilyPolicy:  &singleStack,
					SessionAffinity: v1.ServiceAffinityNone,
					Type:            v1.ServiceTypeClusterIP,
				},
			},
		},
		{
			service: &v1.Service{
				ObjectMeta: om(serviceName),
				Spec: v1.ServiceSpec{
					Ports: []v1.ServicePort{
						{Name: "foo", Port: 443, Protocol: "TCP", TargetPort: intstr.FromInt(6443)},
					},
					Selector:        nil,
					ClusterIP:       "1.2.3.4",
					IPFamilyPolicy:  &singleStack,
					SessionAffinity: v1.ServiceAffinityNone,
					Type:            v1.ServiceTypeLoadBalancer,
				},
			},
			expectUpdate: &v1.Service{
				ObjectMeta: om(serviceName),
				Spec: v1.ServiceSpec{
					Ports: []v1.ServicePort{
						{Name: "https", Port: 443, Protocol: "TCP", TargetPort: intstr.FromInt(6443)},
					},
					Selector:        nil,
					ClusterIP:       "1.2.3.4",
					IPFamilyPolicy:  &singleStack,
					SessionAffinity: v1.ServiceAffinityNone,
					Type:            v1.ServiceTypeClusterIP,
				},
			},
		},
		{
			testName: "service definition satisfies",
			service: &v1.Service{
				ObjectMeta: om(serviceName),
				Spec: v1.ServiceSpec{
					Ports: []v1.ServicePort{
						{Name: "https", Port: 443, Protocol: "TCP", TargetPort: intstr.FromInt(6443)},
					},
					Selector:        nil,
					ClusterIP:       "1.2.3.4",
					IPFamilyPolicy:  &singleStack,
					SessionAffinity: v1.ServiceAffinityNone,
					Type:            v1.ServiceTypeClusterIP,
				},
			},
			expectUpdate: nil,
		},
	}
	for _, test := range reconcileTests {
		t.Run(test.testName, func(t *testing.T) {

			fakeClient := fake.NewSimpleClientset(test.service)

			controller, err := NewController(
				netutils.ParseIPSloppy("1.2.3.4"),
				443,
				0,
				netutils.ParseIPSloppy("192.168.0.1"),
				6443,
				reconcilers.NewNoneEndpointReconciler(),
				0*time.Second,
				reconcilers.NoneEndpointReconcilerType,
				fakeClient)
			if err != nil {
				t.Fatal(err)
			}
			fakeClient.PrependReactor("create", "services", func(action k8stesting.Action) (bool, runtime.Object, error) {
				create := action.(k8stesting.CreateAction)
				controller.serviceInformer.GetIndexer().Add(create.GetObject())
				return true, create.GetObject(), nil
			})
			err = controller.serviceInformer.GetIndexer().Add(test.service)
			if err != nil {
				t.Fatalf("case %q: unexpected error: %v", test.testName, err)
			}

			err = controller.createOrUpdateMasterServiceIfNeeded()
			if err != nil {
				t.Errorf("case %q: unexpected error: %v", test.testName, err)
			}
			updates := []k8stesting.UpdateAction{}
			for _, action := range fakeClient.Actions() {
				if action.GetVerb() == "update" {
					updates = append(updates, action.(k8stesting.UpdateAction))
				}
			}
			if test.expectUpdate != nil {
				if len(updates) != 1 {
					t.Errorf("case %q: unexpected updates: %v", test.testName, updates)
				} else {
					obj := updates[0].GetObject()
					if e, a := test.expectUpdate.ObjectMeta, obj.(*v1.Service).ObjectMeta; !reflect.DeepEqual(e, a) {
						t.Errorf("case %q: expected create:\n%#v\ngot:\n%#v\n", test.testName, e, a)
					}
					if e, a := test.expectUpdate.Spec, obj.(*v1.Service).Spec; !reflect.DeepEqual(e, a) {
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

func TestControllerReconcile(t *testing.T) {
	defaulNs := &v1.Namespace{
		ObjectMeta: metav1.ObjectMeta{Name: metav1.NamespaceDefault},
	}
	defaultSvc := &v1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: metav1.NamespaceDefault,
			Name:      serviceName,
			Labels:    map[string]string{"component": "apiserver", "provider": "kubernetes"},
		},
		Spec: v1.ServiceSpec{
			Ports: []v1.ServicePort{
				{Name: "https", Port: 443, Protocol: "TCP", TargetPort: intstr.FromInt(6443)},
			},
			Selector:        nil,
			ClusterIP:       "1.2.3.4",
			SessionAffinity: v1.ServiceAffinityNone,
			Type:            v1.ServiceTypeClusterIP,
		},
	}

	svc2 := defaultSvc.DeepCopy()
	svc2.Spec.ClusterIP = "4.3.2.1"
	testCases := []struct {
		name    string
		ns      *v1.Namespace
		svc     *v1.Service
		actions [][]string // verb and resource
	}{
		{
			name: "no service no namespace",
			actions: [][]string{
				{"get", "namespaces"},
				{"create", "namespaces"},
				{"create", "services"},
			},
		},
		{
			name: "no service existing namespace",
			ns:   defaulNs,
			actions: [][]string{
				{"get", "namespaces"},
				{"create", "services"},
			},
		},
		{
			name: "existing service and namespace",
			ns:   defaulNs,
			svc:  defaultSvc,
			actions: [][]string{
				{"get", "namespaces"},
			},
		},
		{
			name: "existing service need update",
			ns:   defaulNs,
			svc:  svc2,
			actions: [][]string{
				{"get", "namespaces"},
				{"update", "services"},
			},
		},
	}

	for _, tt := range testCases {
		t.Run(tt.name, func(t *testing.T) {
			objs := []runtime.Object{}
			if tt.ns != nil {
				objs = append(objs, tt.ns)
			}
			if tt.svc != nil {
				objs = append(objs, tt.svc)
			}
			client := fake.NewSimpleClientset(objs...)
			controller, err := NewController(
				netutils.ParseIPSloppy("1.2.3.4"),
				443,
				0,
				netutils.ParseIPSloppy("192.168.0.1"),
				6443,
				reconcilers.NewNoneEndpointReconciler(),
				0*time.Second,
				reconcilers.NoneEndpointReconcilerType,
				client)

			if err != nil {
				t.Fatal(err)
			}
			if tt.svc != nil {
				controller.serviceInformer.GetIndexer().Add(tt.svc)
			}
			controller.sync()
			expectAction(t, client.Actions(), tt.actions)

		})
	}
}

func expectAction(t *testing.T, actions []k8stesting.Action, expected [][]string) {
	t.Helper()
	if len(actions) != len(expected) {
		t.Fatalf("Expected at least %d actions, got %d", len(expected), len(actions))
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

func TestNewController(t *testing.T) {

	tests := []struct {
		name                   string
		serviceIP              net.IP
		publicIP               net.IP
		endpointReconcilerType reconcilers.Type
		wantErr                bool
	}{
		{
			name:                   "service and endpoint ipv4",
			serviceIP:              netutils.ParseIPSloppy("1.2.3.4"),
			publicIP:               netutils.ParseIPSloppy("4.3.2.1"),
			endpointReconcilerType: reconcilers.LeaseEndpointReconcilerType,
		},
		{
			name:                   "service and endpoint different IP family",
			serviceIP:              netutils.ParseIPSloppy("1.2.3.4"),
			publicIP:               netutils.ParseIPSloppy("2001:db8::1"),
			endpointReconcilerType: reconcilers.LeaseEndpointReconcilerType,
			wantErr:                true,
		},
		{
			name:                   "service and endpoint different IP family but no endpoint reconciler",
			serviceIP:              netutils.ParseIPSloppy("1.2.3.4"),
			publicIP:               netutils.ParseIPSloppy("2001:db8::1"),
			endpointReconcilerType: reconcilers.NoneEndpointReconcilerType,
			wantErr:                false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := NewController(tt.serviceIP, 443, 0, tt.publicIP, 6443, nil, 0, tt.endpointReconcilerType, fake.NewSimpleClientset())
			if (err != nil) != tt.wantErr {
				t.Errorf("NewController() error = %v, wantErr %v", err, tt.wantErr)
				return
			}

		})
	}
}
