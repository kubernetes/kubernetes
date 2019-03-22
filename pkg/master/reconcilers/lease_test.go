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

package reconcilers

/*
Original Source:
https://github.com/openshift/origin/blob/bb340c5dd5ff72718be86fb194dedc0faed7f4c7/pkg/cmd/server/election/lease_endpoint_reconciler_test.go
*/

import (
	"net"
	"reflect"
	"testing"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes/fake"
)

type fakeLeases struct {
	keys map[string]bool
}

var _ Leases = &fakeLeases{}

func newFakeLeases() *fakeLeases {
	return &fakeLeases{make(map[string]bool)}
}

func (f *fakeLeases) ListLeases() ([]string, error) {
	res := make([]string, 0, len(f.keys))
	for ip := range f.keys {
		res = append(res, ip)
	}
	return res, nil
}

func (f *fakeLeases) UpdateLease(ip string) error {
	f.keys[ip] = true
	return nil
}

func (f *fakeLeases) RemoveLease(ip string) error {
	delete(f.keys, ip)
	return nil
}

func (f *fakeLeases) SetKeys(keys []string) {
	for _, ip := range keys {
		f.keys[ip] = false
	}
}

func (f *fakeLeases) GetUpdatedKeys() []string {
	res := []string{}
	for ip, updated := range f.keys {
		if updated {
			res = append(res, ip)
		}
	}
	return res
}

func TestLeaseEndpointReconciler(t *testing.T) {
	ns := corev1.NamespaceDefault
	om := func(name string) metav1.ObjectMeta {
		return metav1.ObjectMeta{Namespace: ns, Name: name}
	}
	reconcileTests := []struct {
		testName      string
		serviceName   string
		ip            string
		endpointPorts []corev1.EndpointPort
		endpointKeys  []string
		endpoints     *corev1.EndpointsList
		expectUpdate  *corev1.Endpoints // nil means none expected
	}{
		{
			testName:      "no existing endpoints",
			serviceName:   "foo",
			ip:            "1.2.3.4",
			endpointPorts: []corev1.EndpointPort{{Name: "foo", Port: 8080, Protocol: "TCP"}},
			endpoints:     nil,
			expectUpdate: &corev1.Endpoints{
				ObjectMeta: om("foo"),
				Subsets: []corev1.EndpointSubset{{
					Addresses: []corev1.EndpointAddress{{IP: "1.2.3.4"}},
					Ports:     []corev1.EndpointPort{{Name: "foo", Port: 8080, Protocol: "TCP"}},
				}},
			},
		},
		{
			testName:      "existing endpoints satisfy",
			serviceName:   "foo",
			ip:            "1.2.3.4",
			endpointPorts: []corev1.EndpointPort{{Name: "foo", Port: 8080, Protocol: "TCP"}},
			endpoints: &corev1.EndpointsList{
				Items: []corev1.Endpoints{{
					ObjectMeta: om("foo"),
					Subsets: []corev1.EndpointSubset{{
						Addresses: []corev1.EndpointAddress{{IP: "1.2.3.4"}},
						Ports:     []corev1.EndpointPort{{Name: "foo", Port: 8080, Protocol: "TCP"}},
					}},
				}},
			},
		},
		{
			testName:      "existing endpoints satisfy + refresh existing key",
			serviceName:   "foo",
			ip:            "1.2.3.4",
			endpointPorts: []corev1.EndpointPort{{Name: "foo", Port: 8080, Protocol: "TCP"}},
			endpointKeys:  []string{"1.2.3.4"},
			endpoints: &corev1.EndpointsList{
				Items: []corev1.Endpoints{{
					ObjectMeta: om("foo"),
					Subsets: []corev1.EndpointSubset{{
						Addresses: []corev1.EndpointAddress{{IP: "1.2.3.4"}},
						Ports:     []corev1.EndpointPort{{Name: "foo", Port: 8080, Protocol: "TCP"}},
					}},
				}},
			},
		},
		{
			testName:      "existing endpoints satisfy but too many",
			serviceName:   "foo",
			ip:            "1.2.3.4",
			endpointPorts: []corev1.EndpointPort{{Name: "foo", Port: 8080, Protocol: "TCP"}},
			endpoints: &corev1.EndpointsList{
				Items: []corev1.Endpoints{{
					ObjectMeta: om("foo"),
					Subsets: []corev1.EndpointSubset{{
						Addresses: []corev1.EndpointAddress{{IP: "1.2.3.4"}, {IP: "4.3.2.1"}},
						Ports:     []corev1.EndpointPort{{Name: "foo", Port: 8080, Protocol: "TCP"}},
					}},
				}},
			},
			expectUpdate: &corev1.Endpoints{
				ObjectMeta: om("foo"),
				Subsets: []corev1.EndpointSubset{{
					Addresses: []corev1.EndpointAddress{{IP: "1.2.3.4"}},
					Ports:     []corev1.EndpointPort{{Name: "foo", Port: 8080, Protocol: "TCP"}},
				}},
			},
		},
		{
			testName:      "existing endpoints satisfy but too many + extra masters",
			serviceName:   "foo",
			ip:            "1.2.3.4",
			endpointPorts: []corev1.EndpointPort{{Name: "foo", Port: 8080, Protocol: "TCP"}},
			endpointKeys:  []string{"1.2.3.4", "4.3.2.2", "4.3.2.3", "4.3.2.4"},
			endpoints: &corev1.EndpointsList{
				Items: []corev1.Endpoints{{
					ObjectMeta: om("foo"),
					Subsets: []corev1.EndpointSubset{{
						Addresses: []corev1.EndpointAddress{
							{IP: "1.2.3.4"},
							{IP: "4.3.2.1"},
							{IP: "4.3.2.2"},
							{IP: "4.3.2.3"},
							{IP: "4.3.2.4"},
						},
						Ports: []corev1.EndpointPort{{Name: "foo", Port: 8080, Protocol: "TCP"}},
					}},
				}},
			},
			expectUpdate: &corev1.Endpoints{
				ObjectMeta: om("foo"),
				Subsets: []corev1.EndpointSubset{{
					Addresses: []corev1.EndpointAddress{
						{IP: "1.2.3.4"},
						{IP: "4.3.2.2"},
						{IP: "4.3.2.3"},
						{IP: "4.3.2.4"},
					},
					Ports: []corev1.EndpointPort{{Name: "foo", Port: 8080, Protocol: "TCP"}},
				}},
			},
		},
		{
			testName:      "existing endpoints satisfy but too many + extra masters + delete first",
			serviceName:   "foo",
			ip:            "4.3.2.4",
			endpointPorts: []corev1.EndpointPort{{Name: "foo", Port: 8080, Protocol: "TCP"}},
			endpointKeys:  []string{"4.3.2.1", "4.3.2.2", "4.3.2.3", "4.3.2.4"},
			endpoints: &corev1.EndpointsList{
				Items: []corev1.Endpoints{{
					ObjectMeta: om("foo"),
					Subsets: []corev1.EndpointSubset{{
						Addresses: []corev1.EndpointAddress{
							{IP: "1.2.3.4"},
							{IP: "4.3.2.1"},
							{IP: "4.3.2.2"},
							{IP: "4.3.2.3"},
							{IP: "4.3.2.4"},
						},
						Ports: []corev1.EndpointPort{{Name: "foo", Port: 8080, Protocol: "TCP"}},
					}},
				}},
			},
			expectUpdate: &corev1.Endpoints{
				ObjectMeta: om("foo"),
				Subsets: []corev1.EndpointSubset{{
					Addresses: []corev1.EndpointAddress{
						{IP: "4.3.2.1"},
						{IP: "4.3.2.2"},
						{IP: "4.3.2.3"},
						{IP: "4.3.2.4"},
					},
					Ports: []corev1.EndpointPort{{Name: "foo", Port: 8080, Protocol: "TCP"}},
				}},
			},
		},
		{
			testName:      "existing endpoints current IP missing",
			serviceName:   "foo",
			ip:            "4.3.2.2",
			endpointPorts: []corev1.EndpointPort{{Name: "foo", Port: 8080, Protocol: "TCP"}},
			endpointKeys:  []string{"4.3.2.1"},
			endpoints: &corev1.EndpointsList{
				Items: []corev1.Endpoints{{
					ObjectMeta: om("foo"),
					Subsets: []corev1.EndpointSubset{{
						Addresses: []corev1.EndpointAddress{
							{IP: "4.3.2.1"},
						},
						Ports: []corev1.EndpointPort{{Name: "foo", Port: 8080, Protocol: "TCP"}},
					}},
				}},
			},
			expectUpdate: &corev1.Endpoints{
				ObjectMeta: om("foo"),
				Subsets: []corev1.EndpointSubset{{
					Addresses: []corev1.EndpointAddress{
						{IP: "4.3.2.1"},
						{IP: "4.3.2.2"},
					},
					Ports: []corev1.EndpointPort{{Name: "foo", Port: 8080, Protocol: "TCP"}},
				}},
			},
		},
		{
			testName:      "existing endpoints wrong name",
			serviceName:   "foo",
			ip:            "1.2.3.4",
			endpointPorts: []corev1.EndpointPort{{Name: "foo", Port: 8080, Protocol: "TCP"}},
			endpoints: &corev1.EndpointsList{
				Items: []corev1.Endpoints{{
					ObjectMeta: om("bar"),
					Subsets: []corev1.EndpointSubset{{
						Addresses: []corev1.EndpointAddress{{IP: "1.2.3.4"}},
						Ports:     []corev1.EndpointPort{{Name: "foo", Port: 8080, Protocol: "TCP"}},
					}},
				}},
			},
			expectUpdate: &corev1.Endpoints{
				ObjectMeta: om("foo"),
				Subsets: []corev1.EndpointSubset{{
					Addresses: []corev1.EndpointAddress{{IP: "1.2.3.4"}},
					Ports:     []corev1.EndpointPort{{Name: "foo", Port: 8080, Protocol: "TCP"}},
				}},
			},
		},
		{
			testName:      "existing endpoints wrong IP",
			serviceName:   "foo",
			ip:            "1.2.3.4",
			endpointPorts: []corev1.EndpointPort{{Name: "foo", Port: 8080, Protocol: "TCP"}},
			endpoints: &corev1.EndpointsList{
				Items: []corev1.Endpoints{{
					ObjectMeta: om("foo"),
					Subsets: []corev1.EndpointSubset{{
						Addresses: []corev1.EndpointAddress{{IP: "4.3.2.1"}},
						Ports:     []corev1.EndpointPort{{Name: "foo", Port: 8080, Protocol: "TCP"}},
					}},
				}},
			},
			expectUpdate: &corev1.Endpoints{
				ObjectMeta: om("foo"),
				Subsets: []corev1.EndpointSubset{{
					Addresses: []corev1.EndpointAddress{{IP: "1.2.3.4"}},
					Ports:     []corev1.EndpointPort{{Name: "foo", Port: 8080, Protocol: "TCP"}},
				}},
			},
		},
		{
			testName:      "existing endpoints wrong port",
			serviceName:   "foo",
			ip:            "1.2.3.4",
			endpointPorts: []corev1.EndpointPort{{Name: "foo", Port: 8080, Protocol: "TCP"}},
			endpoints: &corev1.EndpointsList{
				Items: []corev1.Endpoints{{
					ObjectMeta: om("foo"),
					Subsets: []corev1.EndpointSubset{{
						Addresses: []corev1.EndpointAddress{{IP: "1.2.3.4"}},
						Ports:     []corev1.EndpointPort{{Name: "foo", Port: 9090, Protocol: "TCP"}},
					}},
				}},
			},
			expectUpdate: &corev1.Endpoints{
				ObjectMeta: om("foo"),
				Subsets: []corev1.EndpointSubset{{
					Addresses: []corev1.EndpointAddress{{IP: "1.2.3.4"}},
					Ports:     []corev1.EndpointPort{{Name: "foo", Port: 8080, Protocol: "TCP"}},
				}},
			},
		},
		{
			testName:      "existing endpoints wrong protocol",
			serviceName:   "foo",
			ip:            "1.2.3.4",
			endpointPorts: []corev1.EndpointPort{{Name: "foo", Port: 8080, Protocol: "TCP"}},
			endpoints: &corev1.EndpointsList{
				Items: []corev1.Endpoints{{
					ObjectMeta: om("foo"),
					Subsets: []corev1.EndpointSubset{{
						Addresses: []corev1.EndpointAddress{{IP: "1.2.3.4"}},
						Ports:     []corev1.EndpointPort{{Name: "foo", Port: 8080, Protocol: "UDP"}},
					}},
				}},
			},
			expectUpdate: &corev1.Endpoints{
				ObjectMeta: om("foo"),
				Subsets: []corev1.EndpointSubset{{
					Addresses: []corev1.EndpointAddress{{IP: "1.2.3.4"}},
					Ports:     []corev1.EndpointPort{{Name: "foo", Port: 8080, Protocol: "TCP"}},
				}},
			},
		},
		{
			testName:      "existing endpoints wrong port name",
			serviceName:   "foo",
			ip:            "1.2.3.4",
			endpointPorts: []corev1.EndpointPort{{Name: "baz", Port: 8080, Protocol: "TCP"}},
			endpoints: &corev1.EndpointsList{
				Items: []corev1.Endpoints{{
					ObjectMeta: om("foo"),
					Subsets: []corev1.EndpointSubset{{
						Addresses: []corev1.EndpointAddress{{IP: "1.2.3.4"}},
						Ports:     []corev1.EndpointPort{{Name: "foo", Port: 8080, Protocol: "TCP"}},
					}},
				}},
			},
			expectUpdate: &corev1.Endpoints{
				ObjectMeta: om("foo"),
				Subsets: []corev1.EndpointSubset{{
					Addresses: []corev1.EndpointAddress{{IP: "1.2.3.4"}},
					Ports:     []corev1.EndpointPort{{Name: "baz", Port: 8080, Protocol: "TCP"}},
				}},
			},
		},
		{
			testName:    "existing endpoints extra service ports satisfy",
			serviceName: "foo",
			ip:          "1.2.3.4",
			endpointPorts: []corev1.EndpointPort{
				{Name: "foo", Port: 8080, Protocol: "TCP"},
				{Name: "bar", Port: 1000, Protocol: "TCP"},
				{Name: "baz", Port: 1010, Protocol: "TCP"},
			},
			endpoints: &corev1.EndpointsList{
				Items: []corev1.Endpoints{{
					ObjectMeta: om("foo"),
					Subsets: []corev1.EndpointSubset{{
						Addresses: []corev1.EndpointAddress{{IP: "1.2.3.4"}},
						Ports: []corev1.EndpointPort{
							{Name: "foo", Port: 8080, Protocol: "TCP"},
							{Name: "bar", Port: 1000, Protocol: "TCP"},
							{Name: "baz", Port: 1010, Protocol: "TCP"},
						},
					}},
				}},
			},
		},
		{
			testName:    "existing endpoints extra service ports missing port",
			serviceName: "foo",
			ip:          "1.2.3.4",
			endpointPorts: []corev1.EndpointPort{
				{Name: "foo", Port: 8080, Protocol: "TCP"},
				{Name: "bar", Port: 1000, Protocol: "TCP"},
			},
			endpoints: &corev1.EndpointsList{
				Items: []corev1.Endpoints{{
					ObjectMeta: om("foo"),
					Subsets: []corev1.EndpointSubset{{
						Addresses: []corev1.EndpointAddress{{IP: "1.2.3.4"}},
						Ports:     []corev1.EndpointPort{{Name: "foo", Port: 8080, Protocol: "TCP"}},
					}},
				}},
			},
			expectUpdate: &corev1.Endpoints{
				ObjectMeta: om("foo"),
				Subsets: []corev1.EndpointSubset{{
					Addresses: []corev1.EndpointAddress{{IP: "1.2.3.4"}},
					Ports: []corev1.EndpointPort{
						{Name: "foo", Port: 8080, Protocol: "TCP"},
						{Name: "bar", Port: 1000, Protocol: "TCP"},
					},
				}},
			},
		},
	}
	for _, test := range reconcileTests {
		fakeLeases := newFakeLeases()
		fakeLeases.SetKeys(test.endpointKeys)
		clientset := fake.NewSimpleClientset()
		if test.endpoints != nil {
			for _, ep := range test.endpoints.Items {
				if _, err := clientset.CoreV1().Endpoints(ep.Namespace).Create(&ep); err != nil {
					t.Errorf("case %q: unexpected error: %v", test.testName, err)
					continue
				}
			}
		}
		r := NewLeaseEndpointReconciler(clientset.CoreV1(), fakeLeases)
		err := r.ReconcileEndpoints(test.serviceName, net.ParseIP(test.ip), test.endpointPorts, true)
		if err != nil {
			t.Errorf("case %q: unexpected error: %v", test.testName, err)
		}
		actualEndpoints, err := clientset.CoreV1().Endpoints(corev1.NamespaceDefault).Get(test.serviceName, metav1.GetOptions{})
		if err != nil {
			t.Errorf("case %q: unexpected error: %v", test.testName, err)
		}
		if test.expectUpdate != nil {
			if e, a := test.expectUpdate, actualEndpoints; !reflect.DeepEqual(e, a) {
				t.Errorf("case %q: expected update:\n%#v\ngot:\n%#v\n", test.testName, e, a)
			}
		}
		if updatedKeys := fakeLeases.GetUpdatedKeys(); len(updatedKeys) != 1 || updatedKeys[0] != test.ip {
			t.Errorf("case %q: expected the master's IP to be refreshed, but the following IPs were refreshed instead: %v", test.testName, updatedKeys)
		}
	}

	nonReconcileTests := []struct {
		testName      string
		serviceName   string
		ip            string
		endpointPorts []corev1.EndpointPort
		endpointKeys  []string
		endpoints     *corev1.EndpointsList
		expectUpdate  *corev1.Endpoints // nil means none expected
	}{
		{
			testName:    "existing endpoints extra service ports missing port no update",
			serviceName: "foo",
			ip:          "1.2.3.4",
			endpointPorts: []corev1.EndpointPort{
				{Name: "foo", Port: 8080, Protocol: "TCP"},
				{Name: "bar", Port: 1000, Protocol: "TCP"},
			},
			endpoints: &corev1.EndpointsList{
				Items: []corev1.Endpoints{{
					ObjectMeta: om("foo"),
					Subsets: []corev1.EndpointSubset{{
						Addresses: []corev1.EndpointAddress{{IP: "1.2.3.4"}},
						Ports:     []corev1.EndpointPort{{Name: "foo", Port: 8080, Protocol: "TCP"}},
					}},
				}},
			},
			expectUpdate: nil,
		},
		{
			testName:    "existing endpoints extra service ports, wrong ports, wrong IP",
			serviceName: "foo",
			ip:          "1.2.3.4",
			endpointPorts: []corev1.EndpointPort{
				{Name: "foo", Port: 8080, Protocol: "TCP"},
				{Name: "bar", Port: 1000, Protocol: "TCP"},
			},
			endpoints: &corev1.EndpointsList{
				Items: []corev1.Endpoints{{
					ObjectMeta: om("foo"),
					Subsets: []corev1.EndpointSubset{{
						Addresses: []corev1.EndpointAddress{{IP: "4.3.2.1"}},
						Ports:     []corev1.EndpointPort{{Name: "foo", Port: 8080, Protocol: "TCP"}},
					}},
				}},
			},
			expectUpdate: &corev1.Endpoints{
				ObjectMeta: om("foo"),
				Subsets: []corev1.EndpointSubset{{
					Addresses: []corev1.EndpointAddress{{IP: "1.2.3.4"}},
					Ports:     []corev1.EndpointPort{{Name: "foo", Port: 8080, Protocol: "TCP"}},
				}},
			},
		},
		{
			testName:      "no existing endpoints",
			serviceName:   "foo",
			ip:            "1.2.3.4",
			endpointPorts: []corev1.EndpointPort{{Name: "foo", Port: 8080, Protocol: "TCP"}},
			endpoints:     nil,
			expectUpdate: &corev1.Endpoints{
				ObjectMeta: om("foo"),
				Subsets: []corev1.EndpointSubset{{
					Addresses: []corev1.EndpointAddress{{IP: "1.2.3.4"}},
					Ports:     []corev1.EndpointPort{{Name: "foo", Port: 8080, Protocol: "TCP"}},
				}},
			},
		},
	}
	for _, test := range nonReconcileTests {
		t.Run(test.testName, func(t *testing.T) {
			fakeLeases := newFakeLeases()
			fakeLeases.SetKeys(test.endpointKeys)
			clientset := fake.NewSimpleClientset()
			if test.endpoints != nil {
				for _, ep := range test.endpoints.Items {
					if _, err := clientset.CoreV1().Endpoints(ep.Namespace).Create(&ep); err != nil {
						t.Errorf("case %q: unexpected error: %v", test.testName, err)
						continue
					}
				}
			}
			r := NewLeaseEndpointReconciler(clientset.CoreV1(), fakeLeases)
			err := r.ReconcileEndpoints(test.serviceName, net.ParseIP(test.ip), test.endpointPorts, false)
			if err != nil {
				t.Errorf("case %q: unexpected error: %v", test.testName, err)
			}
			actualEndpoints, err := clientset.CoreV1().Endpoints(corev1.NamespaceDefault).Get(test.serviceName, metav1.GetOptions{})
			if err != nil {
				t.Errorf("case %q: unexpected error: %v", test.testName, err)
			}
			if test.expectUpdate != nil {
				if e, a := test.expectUpdate, actualEndpoints; !reflect.DeepEqual(e, a) {
					t.Errorf("case %q: expected update:\n%#v\ngot:\n%#v\n", test.testName, e, a)
				}
			}
			if updatedKeys := fakeLeases.GetUpdatedKeys(); len(updatedKeys) != 1 || updatedKeys[0] != test.ip {
				t.Errorf("case %q: expected the master's IP to be refreshed, but the following IPs were refreshed instead: %v", test.testName, updatedKeys)
			}
		})
	}
}

func TestLeaseRemoveEndpoints(t *testing.T) {
	ns := corev1.NamespaceDefault
	om := func(name string) metav1.ObjectMeta {
		return metav1.ObjectMeta{Namespace: ns, Name: name}
	}
	stopTests := []struct {
		testName      string
		serviceName   string
		ip            string
		endpointPorts []corev1.EndpointPort
		endpointKeys  []string
		endpoints     *corev1.EndpointsList
		expectUpdate  *corev1.Endpoints // nil means none expected
	}{
		{
			testName:      "successful stop reconciling",
			serviceName:   "foo",
			ip:            "1.2.3.4",
			endpointPorts: []corev1.EndpointPort{{Name: "foo", Port: 8080, Protocol: "TCP"}},
			endpointKeys:  []string{"1.2.3.4", "4.3.2.2", "4.3.2.3", "4.3.2.4"},
			endpoints: &corev1.EndpointsList{
				Items: []corev1.Endpoints{{
					ObjectMeta: om("foo"),
					Subsets: []corev1.EndpointSubset{{
						Addresses: []corev1.EndpointAddress{
							{IP: "1.2.3.4"},
							{IP: "4.3.2.2"},
							{IP: "4.3.2.3"},
							{IP: "4.3.2.4"},
						},
						Ports: []corev1.EndpointPort{{Name: "foo", Port: 8080, Protocol: "TCP"}},
					}},
				}},
			},
			expectUpdate: &corev1.Endpoints{
				ObjectMeta: om("foo"),
				Subsets: []corev1.EndpointSubset{{
					Addresses: []corev1.EndpointAddress{
						{IP: "4.3.2.2"},
						{IP: "4.3.2.3"},
						{IP: "4.3.2.4"},
					},
					Ports: []corev1.EndpointPort{{Name: "foo", Port: 8080, Protocol: "TCP"}},
				}},
			},
		},
		{
			testName:      "stop reconciling with ip not in endpoint ip list",
			serviceName:   "foo",
			ip:            "5.6.7.8",
			endpointPorts: []corev1.EndpointPort{{Name: "foo", Port: 8080, Protocol: "TCP"}},
			endpointKeys:  []string{"1.2.3.4", "4.3.2.2", "4.3.2.3", "4.3.2.4"},
			endpoints: &corev1.EndpointsList{
				Items: []corev1.Endpoints{{
					ObjectMeta: om("foo"),
					Subsets: []corev1.EndpointSubset{{
						Addresses: []corev1.EndpointAddress{
							{IP: "1.2.3.4"},
							{IP: "4.3.2.2"},
							{IP: "4.3.2.3"},
							{IP: "4.3.2.4"},
						},
						Ports: []corev1.EndpointPort{{Name: "foo", Port: 8080, Protocol: "TCP"}},
					}},
				}},
			},
		},
	}
	for _, test := range stopTests {
		t.Run(test.testName, func(t *testing.T) {
			fakeLeases := newFakeLeases()
			fakeLeases.SetKeys(test.endpointKeys)
			clientset := fake.NewSimpleClientset()
			for _, ep := range test.endpoints.Items {
				if _, err := clientset.CoreV1().Endpoints(ep.Namespace).Create(&ep); err != nil {
					t.Errorf("case %q: unexpected error: %v", test.testName, err)
					continue
				}
			}
			r := NewLeaseEndpointReconciler(clientset.CoreV1(), fakeLeases)
			err := r.RemoveEndpoints(test.serviceName, net.ParseIP(test.ip), test.endpointPorts)
			if err != nil {
				t.Errorf("case %q: unexpected error: %v", test.testName, err)
			}
			actualEndpoints, err := clientset.CoreV1().Endpoints(corev1.NamespaceDefault).Get(test.serviceName, metav1.GetOptions{})
			if err != nil {
				t.Errorf("case %q: unexpected error: %v", test.testName, err)
			}
			if test.expectUpdate != nil {
				if e, a := test.expectUpdate, actualEndpoints; !reflect.DeepEqual(e, a) {
					t.Errorf("case %q: expected update:\n%#v\ngot:\n%#v\n", test.testName, e, a)
				}
			}
			for _, key := range fakeLeases.GetUpdatedKeys() {
				if key == test.ip {
					t.Errorf("case %q: Found ip %s in leases but shouldn't be there", test.testName, key)
				}
			}
		})
	}
}
