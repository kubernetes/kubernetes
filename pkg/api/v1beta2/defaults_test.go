/*
Copyright 2015 Google Inc. All rights reserved.

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

package v1beta2_test

import (
	"reflect"
	"testing"

	current "github.com/GoogleCloudPlatform/kubernetes/pkg/api/v1beta2"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
)

func roundTrip(t *testing.T, obj runtime.Object) runtime.Object {
	data, err := current.Codec.Encode(obj)
	if err != nil {
		t.Errorf("%v\n %#v", err, obj)
		return nil
	}
	obj2 := reflect.New(reflect.TypeOf(obj).Elem()).Interface().(runtime.Object)
	err = current.Codec.DecodeInto(data, obj2)
	if err != nil {
		t.Errorf("%v\nData: %s\nSource: %#v", err, string(data), obj)
		return nil
	}
	return obj2
}

func TestSetDefaultService(t *testing.T) {
	svc := &current.Service{}
	obj2 := roundTrip(t, runtime.Object(svc))
	svc2 := obj2.(*current.Service)
	if svc2.Protocol != current.ProtocolTCP {
		t.Errorf("Expected default protocol :%s, got: %s", current.ProtocolTCP, svc2.Protocol)
	}
	if svc2.SessionAffinity != current.AffinityTypeNone {
		t.Errorf("Expected default sesseion affinity type:%s, got: %s", current.AffinityTypeNone, svc2.SessionAffinity)
	}
}

func TestSetDefaultSecret(t *testing.T) {
	s := &current.Secret{}
	obj2 := roundTrip(t, runtime.Object(s))
	s2 := obj2.(*current.Secret)

	if s2.Type != current.SecretTypeOpaque {
		t.Errorf("Expected secret type %v, got %v", current.SecretTypeOpaque, s2.Type)
	}
}

func TestSetDefaulEndpointsLegacy(t *testing.T) {
	in := &current.Endpoints{
		Protocol:   "UDP",
		Endpoints:  []string{"1.2.3.4:93", "5.6.7.8:76"},
		TargetRefs: []current.EndpointObjectReference{{Endpoint: "1.2.3.4:93", ObjectReference: current.ObjectReference{ID: "foo"}}},
	}
	obj := roundTrip(t, runtime.Object(in))
	out := obj.(*current.Endpoints)

	if len(out.Subsets) != 2 {
		t.Errorf("Expected 2 EndpointSubsets, got %d (%#v)", len(out.Subsets), out.Subsets)
	}
	expected := []current.EndpointSubset{
		{
			Addresses: []current.EndpointAddress{{IP: "1.2.3.4", TargetRef: &current.ObjectReference{ID: "foo"}}},
			Ports:     []current.EndpointPort{{Protocol: current.ProtocolUDP, Port: 93}},
		},
		{
			Addresses: []current.EndpointAddress{{IP: "5.6.7.8"}},
			Ports:     []current.EndpointPort{{Protocol: current.ProtocolUDP, Port: 76}},
		},
	}
	if !reflect.DeepEqual(out.Subsets, expected) {
		t.Errorf("Expected %#v, got %#v", expected, out.Subsets)
	}
}

func TestSetDefaulEndpointsProtocol(t *testing.T) {
	in := &current.Endpoints{Subsets: []current.EndpointSubset{
		{Ports: []current.EndpointPort{{}, {Protocol: "UDP"}, {}}},
	}}
	obj := roundTrip(t, runtime.Object(in))
	out := obj.(*current.Endpoints)

	if out.Protocol != current.ProtocolTCP {
		t.Errorf("Expected protocol %s, got %s", current.ProtocolTCP, out.Protocol)
	}
	for i := range out.Subsets {
		for j := range out.Subsets[i].Ports {
			if in.Subsets[i].Ports[j].Protocol == "" {
				if out.Subsets[i].Ports[j].Protocol != current.ProtocolTCP {
					t.Errorf("Expected protocol %s, got %s", current.ProtocolTCP, out.Subsets[i].Ports[j].Protocol)
				}
			} else {
				if out.Subsets[i].Ports[j].Protocol != in.Subsets[i].Ports[j].Protocol {
					t.Errorf("Expected protocol %s, got %s", in.Subsets[i].Ports[j].Protocol, out.Subsets[i].Ports[j].Protocol)
				}
			}
		}
	}
}

func TestSetDefaultNamespace(t *testing.T) {
	s := &current.Namespace{}
	obj2 := roundTrip(t, runtime.Object(s))
	s2 := obj2.(*current.Namespace)

	if s2.Status.Phase != current.NamespaceActive {
		t.Errorf("Expected phase %v, got %v", current.NamespaceActive, s2.Status.Phase)
	}
}
