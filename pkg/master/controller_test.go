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

package master

import (
	"net"
	"reflect"
	"testing"
	"time"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/fake"
	"k8s.io/kubernetes/pkg/client/testing/core"
)

var now = time.Now()

const window = 3 * DefaultEndpointReconcilerInterval
const format = "20060102 150405 MST"

type Interface interface{}

func TestDynamicReconciler(t *testing.T) {

	cmTests := []struct {
		testName       string
		cmName         string
		serviceName    string
		ip             string
		endpointPorts  []api.EndpointPort
		reconcilePort  bool
		cm             *api.ConfigMap
		cmExpectUpdate Interface // nil means none expected
		cmExpectCreate Interface // nil means none expected
		ep             *api.EndpointsList
		epExpectUpdate Interface // nil means none expected
		epExpectCreate Interface // nil means none expected
	}{
		{
			testName:      "cm single create",
			cmName:        "apiservers",
			serviceName:   "kubernetes",
			ip:            "1.2.3.4",
			endpointPorts: []api.EndpointPort{{Name: "foo", Port: 8080, Protocol: "TCP"}},
			cmExpectCreate: &api.ConfigMap{
				ObjectMeta: api.ObjectMeta{Name: "apiservers", Namespace: "kube-system"},
				Data:       map[string]string{"1.2.3.4": now.Add(window).Format(format)},
			},
			epExpectCreate: &api.Endpoints{
				ObjectMeta: api.ObjectMeta{Name: "kubernetes", Namespace: "default"},
				Subsets: []api.EndpointSubset{
					{
						Addresses: []api.EndpointAddress{{IP: "1.2.3.4"}},
						Ports:     []api.EndpointPort{{Name: "foo", Port: 8080, Protocol: "TCP"}},
					},
				},
			},
		},

		{
			testName:      "cm single update",
			cmName:        "apiservers",
			serviceName:   "kubernetes",
			ip:            "1.2.3.4",
			endpointPorts: []api.EndpointPort{{Name: "foo", Port: 8080, Protocol: "TCP"}},
			cm: &api.ConfigMap{
				ObjectMeta: api.ObjectMeta{Name: "apiservers", Namespace: "kube-system"},
				Data:       map[string]string{"1.2.3.4": now.Add(-window).Format(format)},
			},
			cmExpectUpdate: &api.ConfigMap{
				ObjectMeta: api.ObjectMeta{Name: "apiservers", Namespace: "kube-system"},
				Data:       map[string]string{"1.2.3.4": now.Add(window).Format(format)},
			},
			epExpectCreate: &api.Endpoints{
				ObjectMeta: api.ObjectMeta{Name: "kubernetes", Namespace: "default"},
				Subsets: []api.EndpointSubset{
					{
						Addresses: []api.EndpointAddress{{IP: "1.2.3.4"}},
						Ports:     []api.EndpointPort{{Name: "foo", Port: 8080, Protocol: "TCP"}},
					},
				},
			},
		},

		{
			testName:      "cm multiple update",
			cmName:        "apiservers",
			serviceName:   "kubernetes",
			ip:            "1.2.3.4",
			endpointPorts: []api.EndpointPort{{Name: "foo", Port: 8080, Protocol: "TCP"}},
			cm: &api.ConfigMap{
				ObjectMeta: api.ObjectMeta{Name: "apiservers", Namespace: "kube-system"},
				Data: map[string]string{
					"1.2.3.4": now.Format(format),
					"1.2.3.5": now.Add(-2 * window).Format(format),
					"1.2.3.6": now.Add(window / 2).Format(format),
					"1.2.3.7": now.Add((3 * window) / 2).Format(format),
				},
			},
			cmExpectUpdate: &api.ConfigMap{
				ObjectMeta: api.ObjectMeta{Name: "apiservers", Namespace: "kube-system"},
				Data: map[string]string{
					"1.2.3.4": now.Add(window).Format(format),
					"1.2.3.6": now.Add(window / 2).Format(format),
					"1.2.3.7": now.Add((3 * window) / 2).Format(format),
				},
			},
			epExpectCreate: &api.Endpoints{
				ObjectMeta: api.ObjectMeta{Name: "kubernetes", Namespace: "default"},
				Subsets: []api.EndpointSubset{
					{
						Addresses: []api.EndpointAddress{
							{IP: "1.2.3.4"},
							{IP: "1.2.3.6"},
							{IP: "1.2.3.7"},
						},
						Ports: []api.EndpointPort{{Name: "foo", Port: 8080, Protocol: "TCP"}},
					},
				},
			},
		},

		{
			testName:      "ep multiple update",
			cmName:        "apiservers",
			serviceName:   "kubernetes",
			ip:            "1.2.3.4",
			endpointPorts: []api.EndpointPort{{Name: "foo", Port: 8080, Protocol: "TCP"}},
			cm: &api.ConfigMap{
				ObjectMeta: api.ObjectMeta{Name: "apiservers", Namespace: "kube-system"},
				Data: map[string]string{
					"1.2.3.4": now.Format(format),
					"1.2.3.5": now.Add(-2 * window).Format(format),
					"1.2.3.6": now.Add(window / 2).Format(format),
					"1.2.3.7": now.Add((3 * window) / 2).Format(format),
				},
			},
			cmExpectUpdate: &api.ConfigMap{
				ObjectMeta: api.ObjectMeta{Name: "apiservers", Namespace: "kube-system"},
				Data: map[string]string{
					"1.2.3.4": now.Add(window).Format(format),
					"1.2.3.6": now.Add(window / 2).Format(format),
					"1.2.3.7": now.Add((3 * window) / 2).Format(format),
				},
			},
			ep: &api.EndpointsList{
				Items: []api.Endpoints{
					{
						ObjectMeta: api.ObjectMeta{Name: "kubernetes", Namespace: "default"},
						Subsets: []api.EndpointSubset{
							{
								Addresses: []api.EndpointAddress{
									{IP: "1.2.3.4"},
									{IP: "1.2.3.5"},
									{IP: "1.2.3.7"},
								},
								Ports: []api.EndpointPort{{Name: "foo", Port: 8080, Protocol: "TCP"}},
							},
						},
					},
				},
			},
			epExpectUpdate: &api.Endpoints{
				ObjectMeta: api.ObjectMeta{Name: "kubernetes", Namespace: "default"},
				Subsets: []api.EndpointSubset{
					{
						Addresses: []api.EndpointAddress{
							{IP: "1.2.3.4"},
							{IP: "1.2.3.6"},
							{IP: "1.2.3.7"},
						},
						Ports: []api.EndpointPort{{Name: "foo", Port: 8080, Protocol: "TCP"}},
					},
				},
			},
		},

		{
			testName:      "ep do reconcile port",
			cmName:        "apiservers",
			serviceName:   "kubernetes",
			ip:            "1.2.3.4",
			endpointPorts: []api.EndpointPort{{Name: "bar", Port: 9090, Protocol: "TCP"}},
			reconcilePort: true,
			cm: &api.ConfigMap{
				ObjectMeta: api.ObjectMeta{Name: "apiservers", Namespace: "kube-system"},
				Data: map[string]string{
					"1.2.3.4": now.Add(window / 2).Format(format),
					"1.2.3.5": now.Add(window / 2).Format(format),
				},
			},
			cmExpectUpdate: &api.ConfigMap{
				ObjectMeta: api.ObjectMeta{Name: "apiservers", Namespace: "kube-system"},
				Data: map[string]string{
					"1.2.3.4": now.Add(window).Format(format),
					"1.2.3.5": now.Add(window / 2).Format(format),
				},
			},
			ep: &api.EndpointsList{
				Items: []api.Endpoints{
					{
						ObjectMeta: api.ObjectMeta{Name: "kubernetes", Namespace: "default"},
						Subsets: []api.EndpointSubset{
							{
								Addresses: []api.EndpointAddress{
									{IP: "1.2.3.4"},
									{IP: "1.2.3.5"},
								},
								Ports: []api.EndpointPort{{Name: "foo", Port: 8080, Protocol: "TCP"}},
							},
						},
					},
				},
			},
			epExpectUpdate: &api.Endpoints{
				ObjectMeta: api.ObjectMeta{Name: "kubernetes", Namespace: "default"},
				Subsets: []api.EndpointSubset{
					{
						Addresses: []api.EndpointAddress{
							{IP: "1.2.3.4"},
							{IP: "1.2.3.5"},
						},
						Ports: []api.EndpointPort{{Name: "bar", Port: 9090, Protocol: "TCP"}},
					},
				},
			},
		},

		{
			testName:      "ep don't reconcile port",
			cmName:        "apiservers",
			serviceName:   "kubernetes",
			ip:            "1.2.3.4",
			endpointPorts: []api.EndpointPort{{Name: "bar", Port: 9090, Protocol: "TCP"}},
			reconcilePort: false,
			cm: &api.ConfigMap{
				ObjectMeta: api.ObjectMeta{Name: "apiservers", Namespace: "kube-system"},
				Data: map[string]string{
					"1.2.3.4": now.Add(window / 2).Format(format),
					"1.2.3.5": now.Add(window / 2).Format(format),
				},
			},
			cmExpectUpdate: &api.ConfigMap{
				ObjectMeta: api.ObjectMeta{Name: "apiservers", Namespace: "kube-system"},
				Data: map[string]string{
					"1.2.3.4": now.Add(window).Format(format),
					"1.2.3.5": now.Add(window / 2).Format(format),
				},
			},
			ep: &api.EndpointsList{
				Items: []api.Endpoints{
					{
						ObjectMeta: api.ObjectMeta{Name: "kubernetes", Namespace: "default"},
						Subsets: []api.EndpointSubset{
							{
								Addresses: []api.EndpointAddress{
									{IP: "1.2.3.4"},
									{IP: "1.2.3.5"},
								},
								Ports: []api.EndpointPort{{Name: "foo", Port: 8080, Protocol: "TCP"}},
							},
						},
					},
				},
			},
		},
	}

	for _, test := range cmTests {

		fakeCMClient := fake.NewSimpleClientset()
		if test.cm != nil {
			fakeCMClient = fake.NewSimpleClientset(test.cm)
		}

		fakeEPClient := fake.NewSimpleClientset()
		if test.ep != nil {
			fakeEPClient = fake.NewSimpleClientset(test.ep)
		}
		reconciler := NewDynamicEndpointReconciler(fakeEPClient.Core(), fakeCMClient.Core())
		err := reconciler.ReconcileEndpoints(test.cmName, test.serviceName, net.ParseIP(test.ip), test.endpointPorts, test.reconcilePort, now)
		if err != nil {
			t.Errorf("case %q: unexpected error: %v", test.testName, err)
		}

		// handle config map
		cmCreates := []core.CreateAction{}
		for _, action := range fakeCMClient.Actions() {
			if action.GetVerb() != "create" {
				continue
			}
			cmCreates = append(cmCreates, action.(core.CreateAction))
		}
		if test.cmExpectCreate != nil {
			if len(cmCreates) == 0 {
				t.Errorf("case %q: unexpected creates: %v", test.testName, test.cmExpectCreate)
			} else if e, a := test.cmExpectCreate, cmCreates[0].GetObject(); !reflect.DeepEqual(e, a) { // compare the first create
				t.Errorf("case %q: expected create:\n%#v\ngot:\n%#v\n", test.testName, e, a)
			}
		}
		if test.cmExpectCreate == nil && len(cmCreates) > 0 {
			t.Errorf("case %q: no create expected, yet saw: %v", test.testName, cmCreates)
		}

		cmUpdates := []core.UpdateAction{}
		for _, action := range fakeCMClient.Actions() {
			if action.GetVerb() != "update" {
				continue
			}
			cmUpdates = append(cmUpdates, action.(core.UpdateAction))
		}
		if test.cmExpectUpdate != nil {
			if len(cmUpdates) == 0 {
				t.Errorf("case %q: unexpected updates: %v", test.testName, cmUpdates)
			} else if e, a := test.cmExpectUpdate, cmUpdates[0].GetObject(); !reflect.DeepEqual(e, a) { // compare the first update
				t.Errorf("case %q: expected update:\n%#v\ngot:\n%#v\n", test.testName, e, a)
			}
		}
		if test.cmExpectUpdate == nil && len(cmUpdates) > 0 {
			t.Errorf("case %q: no update expected, yet saw: %v", test.testName, cmUpdates)
		}

		// handle endpoints
		epCreates := []core.CreateAction{}
		for _, action := range fakeEPClient.Actions() {
			if action.GetVerb() != "create" {
				continue
			}
			epCreates = append(epCreates, action.(core.CreateAction))
		}
		if test.epExpectCreate != nil {
			if len(epCreates) == 0 {
				t.Errorf("case %q: unexpected creates: %v", test.testName, test.epExpectCreate)
			} else if e, a := test.epExpectCreate, epCreates[0].GetObject(); !reflect.DeepEqual(e, a) { // compare the first create
				t.Errorf("case %q: expected create:\n%#v\ngot:\n%#v\n", test.testName, e, a)
			}
		}
		if test.epExpectCreate == nil && len(epCreates) > 0 {
			t.Errorf("case %q: no create expected, yet saw: %v", test.testName, cmCreates)
		}

		epUpdates := []core.UpdateAction{}
		for _, action := range fakeEPClient.Actions() {
			if action.GetVerb() != "update" {
				continue
			}
			epUpdates = append(epUpdates, action.(core.UpdateAction))
		}
		if test.epExpectUpdate != nil {
			if len(epUpdates) == 0 {
				t.Errorf("case %q: unexpected updates: %v", test.testName, epUpdates)
			} else if e, a := test.epExpectUpdate, epUpdates[0].GetObject(); !reflect.DeepEqual(e, a) { // compare the first update
				t.Errorf("case %q: expected update:\n%#v\ngot:\n%#v\n", test.testName, e, a)
			}
		}
		if test.epExpectUpdate == nil && len(epUpdates) > 0 {
			t.Errorf("case %q: no update expected, yet saw: %v", test.testName, epUpdates)
		}
	}
}
