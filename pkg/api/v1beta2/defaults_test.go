/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	versioned "github.com/GoogleCloudPlatform/kubernetes/pkg/api/v1beta2"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
)

func roundTrip(t *testing.T, obj runtime.Object) runtime.Object {
	data, err := versioned.Codec.Encode(obj)
	if err != nil {
		t.Errorf("%v\n %#v", err, obj)
		return nil
	}
	obj2, err := api.Codec.Decode(data)
	if err != nil {
		t.Errorf("%v\nData: %s\nSource: %#v", err, string(data), obj)
		return nil
	}
	obj3 := reflect.New(reflect.TypeOf(obj).Elem()).Interface().(runtime.Object)
	err = api.Scheme.Convert(obj2, obj3)
	if err != nil {
		t.Errorf("%v\nSource: %#v", err, obj2)
		return nil
	}
	return obj3
}

func TestSetDefaultReplicationController(t *testing.T) {
	tests := []struct {
		rc             *versioned.ReplicationController
		expectLabels   bool
		expectSelector bool
	}{
		{
			rc: &versioned.ReplicationController{
				DesiredState: versioned.ReplicationControllerState{
					PodTemplate: versioned.PodTemplate{
						Labels: map[string]string{
							"foo": "bar",
						},
					},
				},
			},
			expectLabels:   true,
			expectSelector: true,
		},
		{
			rc: &versioned.ReplicationController{
				Labels: map[string]string{
					"bar": "foo",
				},
				DesiredState: versioned.ReplicationControllerState{
					PodTemplate: versioned.PodTemplate{
						Labels: map[string]string{
							"foo": "bar",
						},
					},
				},
			},
			expectLabels:   false,
			expectSelector: true,
		},
		{
			rc: &versioned.ReplicationController{
				Labels: map[string]string{
					"bar": "foo",
				},
				DesiredState: versioned.ReplicationControllerState{
					ReplicaSelector: map[string]string{
						"some": "other",
					},
					PodTemplate: versioned.PodTemplate{
						Labels: map[string]string{
							"foo": "bar",
						},
					},
				},
			},
			expectLabels:   false,
			expectSelector: false,
		},
		{
			rc: &versioned.ReplicationController{
				DesiredState: versioned.ReplicationControllerState{
					ReplicaSelector: map[string]string{
						"some": "other",
					},
					PodTemplate: versioned.PodTemplate{
						Labels: map[string]string{
							"foo": "bar",
						},
					},
				},
			},
			expectLabels:   true,
			expectSelector: false,
		},
	}
	for _, test := range tests {
		rc := test.rc
		obj2 := roundTrip(t, runtime.Object(rc))
		rc2, ok := obj2.(*versioned.ReplicationController)
		if !ok {
			t.Errorf("unexpected object: %v", rc2)
			t.FailNow()
		}
		if test.expectSelector != reflect.DeepEqual(rc2.DesiredState.ReplicaSelector, rc2.DesiredState.PodTemplate.Labels) {
			if test.expectSelector {
				t.Errorf("expected: %v, got: %v", rc2.DesiredState.PodTemplate.Labels, rc2.DesiredState.ReplicaSelector)
			} else {
				t.Errorf("unexpected equality: %v", rc2.DesiredState.PodTemplate.Labels)
			}
		}
		if test.expectLabels != reflect.DeepEqual(rc2.Labels, rc2.DesiredState.PodTemplate.Labels) {
			if test.expectLabels {
				t.Errorf("expected: %v, got: %v", rc2.DesiredState.PodTemplate.Labels, rc2.Labels)
			} else {
				t.Errorf("unexpected equality: %v", rc2.DesiredState.PodTemplate.Labels)
			}
		}
	}
}

func TestSetDefaultService(t *testing.T) {
	svc := &versioned.Service{}
	obj2 := roundTrip(t, runtime.Object(svc))
	svc2 := obj2.(*versioned.Service)
	if svc2.Protocol != versioned.ProtocolTCP {
		t.Errorf("Expected default protocol :%s, got: %s", versioned.ProtocolTCP, svc2.Protocol)
	}
	if svc2.SessionAffinity != versioned.ServiceAffinityNone {
		t.Errorf("Expected default session affinity type:%s, got: %s", versioned.ServiceAffinityNone, svc2.SessionAffinity)
	}
	if svc2.Type != versioned.ServiceTypeClusterIP {
		t.Errorf("Expected default type:%s, got: %s", versioned.ServiceTypeClusterIP, svc2.Type)
	}
}

func TestSetDefaultServiceWithLoadbalancer(t *testing.T) {
	svc := &versioned.Service{}
	svc.CreateExternalLoadBalancer = true
	obj2 := roundTrip(t, runtime.Object(svc))
	svc2 := obj2.(*versioned.Service)
	if svc2.Type != versioned.ServiceTypeLoadBalancer {
		t.Errorf("Expected default type:%s, got: %s", versioned.ServiceTypeLoadBalancer, svc2.Type)
	}
}

func TestSetDefaultPersistentVolume(t *testing.T) {
	pv := &versioned.PersistentVolume{}
	obj2 := roundTrip(t, runtime.Object(pv))
	pv2 := obj2.(*versioned.PersistentVolume)

	if pv2.Status.Phase != versioned.VolumePending {
		t.Errorf("Expected volume phase %v, got %v", versioned.VolumePending, pv2.Status.Phase)
	}
}

func TestSetDefaultPersistentVolumeClaim(t *testing.T) {
	pvc := &versioned.PersistentVolumeClaim{}
	obj2 := roundTrip(t, runtime.Object(pvc))
	pvc2 := obj2.(*versioned.PersistentVolumeClaim)

	if pvc2.Status.Phase != versioned.ClaimPending {
		t.Errorf("Expected claim phase %v, got %v", versioned.ClaimPending, pvc2.Status.Phase)
	}
}

func TestSetDefaultSecret(t *testing.T) {
	s := &versioned.Secret{}
	obj2 := roundTrip(t, runtime.Object(s))
	s2 := obj2.(*versioned.Secret)

	if s2.Type != versioned.SecretTypeOpaque {
		t.Errorf("Expected secret type %v, got %v", versioned.SecretTypeOpaque, s2.Type)
	}
}

func TestSetDefaulEndpointsLegacy(t *testing.T) {
	in := &versioned.Endpoints{
		Protocol:   "UDP",
		Endpoints:  []string{"1.2.3.4:93", "5.6.7.8:76"},
		TargetRefs: []versioned.EndpointObjectReference{{Endpoint: "1.2.3.4:93", ObjectReference: versioned.ObjectReference{ID: "foo"}}},
	}
	obj := roundTrip(t, runtime.Object(in))
	out := obj.(*versioned.Endpoints)

	if len(out.Subsets) != 2 {
		t.Errorf("Expected 2 EndpointSubsets, got %d (%#v)", len(out.Subsets), out.Subsets)
	}
	expected := []versioned.EndpointSubset{
		{
			Addresses: []versioned.EndpointAddress{{IP: "1.2.3.4", TargetRef: &versioned.ObjectReference{ID: "foo"}}},
			Ports:     []versioned.EndpointPort{{Protocol: versioned.ProtocolUDP, Port: 93}},
		},
		{
			Addresses: []versioned.EndpointAddress{{IP: "5.6.7.8"}},
			Ports:     []versioned.EndpointPort{{Protocol: versioned.ProtocolUDP, Port: 76}},
		},
	}
	if !reflect.DeepEqual(out.Subsets, expected) {
		t.Errorf("Expected %#v, got %#v", expected, out.Subsets)
	}
}

func TestSetDefaulEndpointsProtocol(t *testing.T) {
	in := &versioned.Endpoints{Subsets: []versioned.EndpointSubset{
		{Ports: []versioned.EndpointPort{{}, {Protocol: "UDP"}, {}}},
	}}
	obj := roundTrip(t, runtime.Object(in))
	out := obj.(*versioned.Endpoints)

	if out.Protocol != versioned.ProtocolTCP {
		t.Errorf("Expected protocol %s, got %s", versioned.ProtocolTCP, out.Protocol)
	}
	for i := range out.Subsets {
		for j := range out.Subsets[i].Ports {
			if in.Subsets[i].Ports[j].Protocol == "" {
				if out.Subsets[i].Ports[j].Protocol != versioned.ProtocolTCP {
					t.Errorf("Expected protocol %s, got %s", versioned.ProtocolTCP, out.Subsets[i].Ports[j].Protocol)
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
	s := &versioned.Namespace{}
	obj2 := roundTrip(t, runtime.Object(s))
	s2 := obj2.(*versioned.Namespace)

	if s2.Status.Phase != versioned.NamespaceActive {
		t.Errorf("Expected phase %v, got %v", versioned.NamespaceActive, s2.Status.Phase)
	}
}

func TestSetDefaultContainerManifestHostNetwork(t *testing.T) {
	portNum := 8080
	s := versioned.ContainerManifest{}
	s.HostNetwork = true
	s.Containers = []versioned.Container{
		{
			Ports: []versioned.ContainerPort{
				{
					ContainerPort: portNum,
				},
			},
		},
	}
	obj2 := roundTrip(t, runtime.Object(&versioned.ContainerManifestList{
		Items: []versioned.ContainerManifest{s},
	}))
	sList2 := obj2.(*versioned.ContainerManifestList)
	s2 := sList2.Items[0]

	hostPortNum := s2.Containers[0].Ports[0].HostPort
	if hostPortNum != portNum {
		t.Errorf("Expected container port to be defaulted, was made %d instead of %d", hostPortNum, portNum)
	}
}

func TestSetDefaultServicePort(t *testing.T) {
	// Unchanged if set.
	in := &versioned.Service{Ports: []versioned.ServicePort{{Protocol: "UDP", Port: 9376, ContainerPort: util.NewIntOrStringFromInt(118)}}}
	out := roundTrip(t, runtime.Object(in)).(*versioned.Service)
	if out.Ports[0].Protocol != versioned.ProtocolUDP {
		t.Errorf("Expected protocol %s, got %s", versioned.ProtocolUDP, out.Ports[0].Protocol)
	}
	if out.Ports[0].ContainerPort != in.Ports[0].ContainerPort {
		t.Errorf("Expected port %d, got %d", in.Ports[0].ContainerPort, out.Ports[0].ContainerPort)
	}

	// Defaulted.
	in = &versioned.Service{Ports: []versioned.ServicePort{{Protocol: "", Port: 9376, ContainerPort: util.NewIntOrStringFromInt(0)}}}
	out = roundTrip(t, runtime.Object(in)).(*versioned.Service)
	if out.Ports[0].Protocol != versioned.ProtocolTCP {
		t.Errorf("Expected protocol %s, got %s", versioned.ProtocolTCP, out.Ports[0].Protocol)
	}
	if out.Ports[0].ContainerPort != util.NewIntOrStringFromInt(in.Ports[0].Port) {
		t.Errorf("Expected port %d, got %v", in.Ports[0].Port, out.Ports[0].ContainerPort)
	}

	// Defaulted.
	in = &versioned.Service{Ports: []versioned.ServicePort{{Protocol: "", Port: 9376, ContainerPort: util.NewIntOrStringFromString("")}}}
	out = roundTrip(t, runtime.Object(in)).(*versioned.Service)
	if out.Ports[0].Protocol != versioned.ProtocolTCP {
		t.Errorf("Expected protocol %s, got %s", versioned.ProtocolTCP, out.Ports[0].Protocol)
	}
	if out.Ports[0].ContainerPort != util.NewIntOrStringFromInt(in.Ports[0].Port) {
		t.Errorf("Expected port %d, got %v", in.Ports[0].Port, out.Ports[0].ContainerPort)
	}
}

func TestSetDefaultMinionExternalID(t *testing.T) {
	name := "node0"
	m := &versioned.Minion{}
	m.ID = name
	obj2 := roundTrip(t, runtime.Object(m))
	m2 := obj2.(*versioned.Minion)
	if m2.ExternalID != name {
		t.Errorf("Expected default External ID: %s, got: %s", name, m2.ExternalID)
	}
}

func TestSetDefaultObjectFieldSelectorAPIVersion(t *testing.T) {
	s := versioned.ContainerManifest{
		Containers: []versioned.Container{
			{
				Env: []versioned.EnvVar{
					{
						ValueFrom: &versioned.EnvVarSource{
							FieldRef: &versioned.ObjectFieldSelector{},
						},
					},
				},
			},
		},
	}
	obj2 := roundTrip(t, runtime.Object(&versioned.ContainerManifestList{
		Items: []versioned.ContainerManifest{s},
	}))
	sList2 := obj2.(*versioned.ContainerManifestList)
	s2 := sList2.Items[0]

	apiVersion := s2.Containers[0].Env[0].ValueFrom.FieldRef.APIVersion
	if apiVersion != "v1beta2" {
		t.Errorf("Expected default APIVersion v1beta2, got: %v", apiVersion)
	}
}

func TestSetDefaultSecurityContext(t *testing.T) {
	priv := false
	privTrue := true
	testCases := map[string]struct {
		c versioned.Container
	}{
		"downward defaulting caps": {
			c: versioned.Container{
				Privileged: false,
				Capabilities: versioned.Capabilities{
					Add:  []versioned.Capability{"foo"},
					Drop: []versioned.Capability{"bar"},
				},
				SecurityContext: &versioned.SecurityContext{
					Privileged: &priv,
				},
			},
		},
		"downward defaulting priv": {
			c: versioned.Container{
				Privileged: false,
				Capabilities: versioned.Capabilities{
					Add:  []versioned.Capability{"foo"},
					Drop: []versioned.Capability{"bar"},
				},
				SecurityContext: &versioned.SecurityContext{
					Capabilities: &versioned.Capabilities{
						Add:  []versioned.Capability{"foo"},
						Drop: []versioned.Capability{"bar"},
					},
				},
			},
		},
		"upward defaulting caps": {
			c: versioned.Container{
				Privileged: false,
				SecurityContext: &versioned.SecurityContext{
					Privileged: &priv,
					Capabilities: &versioned.Capabilities{
						Add:  []versioned.Capability{"biz"},
						Drop: []versioned.Capability{"baz"},
					},
				},
			},
		},
		"upward defaulting priv": {
			c: versioned.Container{
				Capabilities: versioned.Capabilities{
					Add:  []versioned.Capability{"foo"},
					Drop: []versioned.Capability{"bar"},
				},
				SecurityContext: &versioned.SecurityContext{
					Privileged: &privTrue,
					Capabilities: &versioned.Capabilities{
						Add:  []versioned.Capability{"foo"},
						Drop: []versioned.Capability{"bar"},
					},
				},
			},
		},
	}

	pod := &versioned.Pod{
		DesiredState: versioned.PodState{
			Manifest: versioned.ContainerManifest{},
		},
	}

	for k, v := range testCases {
		pod.DesiredState.Manifest.Containers = []versioned.Container{v.c}
		obj := roundTrip(t, runtime.Object(pod))
		defaultedPod := obj.(*versioned.Pod)
		c := defaultedPod.DesiredState.Manifest.Containers[0]
		if isEqual, issues := areSecurityContextAndContainerEqual(&c); !isEqual {
			t.Errorf("test case %s expected the security context to have the same values as the container but found %#v", k, issues)
		}
	}
}

func areSecurityContextAndContainerEqual(c *versioned.Container) (bool, []string) {
	issues := make([]string, 0)
	equal := true

	if c.SecurityContext == nil || c.SecurityContext.Privileged == nil || c.SecurityContext.Capabilities == nil {
		equal = false
		issues = append(issues, "Expected non nil settings for SecurityContext")
		return equal, issues
	}
	if *c.SecurityContext.Privileged != c.Privileged {
		equal = false
		issues = append(issues, "The defaulted SecurityContext.Privileged value did not match the container value")
	}
	if !reflect.DeepEqual(c.Capabilities.Add, c.Capabilities.Add) {
		equal = false
		issues = append(issues, "The defaulted SecurityContext.Capabilities.Add did not match the container settings")
	}
	if !reflect.DeepEqual(c.Capabilities.Drop, c.Capabilities.Drop) {
		equal = false
		issues = append(issues, "The defaulted SecurityContext.Capabilities.Drop did not match the container settings")
	}
	return equal, issues
}
