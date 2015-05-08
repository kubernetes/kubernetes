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

package v1_test

import (
	"reflect"
	"testing"

	newer "github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	current "github.com/GoogleCloudPlatform/kubernetes/pkg/api/v1"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
)

func roundTrip(t *testing.T, obj runtime.Object) runtime.Object {
	data, err := current.Codec.Encode(obj)
	if err != nil {
		t.Errorf("%v\n %#v", err, obj)
		return nil
	}
	obj2, err := newer.Codec.Decode(data)
	if err != nil {
		t.Errorf("%v\nData: %s\nSource: %#v", err, string(data), obj)
		return nil
	}
	obj3 := reflect.New(reflect.TypeOf(obj).Elem()).Interface().(runtime.Object)
	err = newer.Scheme.Convert(obj2, obj3)
	if err != nil {
		t.Errorf("%v\nSource: %#v", err, obj2)
		return nil
	}
	return obj3
}

func TestSetDefaultReplicationController(t *testing.T) {
	tests := []struct {
		rc             *current.ReplicationController
		expectLabels   bool
		expectSelector bool
	}{
		{
			rc: &current.ReplicationController{
				Spec: current.ReplicationControllerSpec{
					Template: &current.PodTemplateSpec{
						ObjectMeta: current.ObjectMeta{
							Labels: map[string]string{
								"foo": "bar",
							},
						},
					},
				},
			},
			expectLabels:   true,
			expectSelector: true,
		},
		{
			rc: &current.ReplicationController{
				ObjectMeta: current.ObjectMeta{
					Labels: map[string]string{
						"bar": "foo",
					},
				},
				Spec: current.ReplicationControllerSpec{
					Template: &current.PodTemplateSpec{
						ObjectMeta: current.ObjectMeta{
							Labels: map[string]string{
								"foo": "bar",
							},
						},
					},
				},
			},
			expectLabels:   false,
			expectSelector: true,
		},
		{
			rc: &current.ReplicationController{
				ObjectMeta: current.ObjectMeta{
					Labels: map[string]string{
						"bar": "foo",
					},
				},
				Spec: current.ReplicationControllerSpec{
					Selector: map[string]string{
						"some": "other",
					},
					Template: &current.PodTemplateSpec{
						ObjectMeta: current.ObjectMeta{
							Labels: map[string]string{
								"foo": "bar",
							},
						},
					},
				},
			},
			expectLabels:   false,
			expectSelector: false,
		},
		{
			rc: &current.ReplicationController{
				Spec: current.ReplicationControllerSpec{
					Selector: map[string]string{
						"some": "other",
					},
					Template: &current.PodTemplateSpec{
						ObjectMeta: current.ObjectMeta{
							Labels: map[string]string{
								"foo": "bar",
							},
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
		rc2, ok := obj2.(*current.ReplicationController)
		if !ok {
			t.Errorf("unexpected object: %v", rc2)
			t.FailNow()
		}
		if test.expectSelector != reflect.DeepEqual(rc2.Spec.Selector, rc2.Spec.Template.Labels) {
			if test.expectSelector {
				t.Errorf("expected: %v, got: %v", rc2.Spec.Template.Labels, rc2.Spec.Selector)
			} else {
				t.Errorf("unexpected equality: %v", rc.Spec.Selector)
			}
		}
		if test.expectLabels != reflect.DeepEqual(rc2.Labels, rc2.Spec.Template.Labels) {
			if test.expectLabels {
				t.Errorf("expected: %v, got: %v", rc2.Spec.Template.Labels, rc2.Labels)
			} else {
				t.Errorf("unexpected equality: %v", rc.Labels)
			}
		}
	}
}

func TestSetDefaultService(t *testing.T) {
	svc := &current.Service{}
	obj2 := roundTrip(t, runtime.Object(svc))
	svc2 := obj2.(*current.Service)
	if svc2.Spec.SessionAffinity != current.AffinityTypeNone {
		t.Errorf("Expected default sesseion affinity type:%s, got: %s", current.AffinityTypeNone, svc2.Spec.SessionAffinity)
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

func TestSetDefaultPersistentVolume(t *testing.T) {
	pv := &current.PersistentVolume{}
	obj2 := roundTrip(t, runtime.Object(pv))
	pv2 := obj2.(*current.PersistentVolume)

	if pv2.Status.Phase != current.VolumePending {
		t.Errorf("Expected volume phase %v, got %v", current.VolumePending, pv2.Status.Phase)
	}
}

func TestSetDefaultPersistentVolumeClaim(t *testing.T) {
	pvc := &current.PersistentVolumeClaim{}
	obj2 := roundTrip(t, runtime.Object(pvc))
	pvc2 := obj2.(*current.PersistentVolumeClaim)

	if pvc2.Status.Phase != current.ClaimPending {
		t.Errorf("Expected claim phase %v, got %v", current.ClaimPending, pvc2.Status.Phase)
	}
}

func TestSetDefaulEndpointsProtocol(t *testing.T) {
	in := &current.Endpoints{Subsets: []current.EndpointSubset{
		{Ports: []current.EndpointPort{{}, {Protocol: "UDP"}, {}}},
	}}
	obj := roundTrip(t, runtime.Object(in))
	out := obj.(*current.Endpoints)

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

func TestSetDefaulServiceTargetPort(t *testing.T) {
	in := &current.Service{Spec: current.ServiceSpec{Ports: []current.ServicePort{{Port: 1234}}}}
	obj := roundTrip(t, runtime.Object(in))
	out := obj.(*current.Service)
	if out.Spec.Ports[0].TargetPort != util.NewIntOrStringFromInt(1234) {
		t.Errorf("Expected TargetPort to be defaulted, got %s", out.Spec.Ports[0].TargetPort)
	}

	in = &current.Service{Spec: current.ServiceSpec{Ports: []current.ServicePort{{Port: 1234, TargetPort: util.NewIntOrStringFromInt(5678)}}}}
	obj = roundTrip(t, runtime.Object(in))
	out = obj.(*current.Service)
	if out.Spec.Ports[0].TargetPort != util.NewIntOrStringFromInt(5678) {
		t.Errorf("Expected TargetPort to be unchanged, got %s", out.Spec.Ports[0].TargetPort)
	}
}

func TestSetDefaultServicePort(t *testing.T) {
	// Unchanged if set.
	in := &current.Service{Spec: current.ServiceSpec{
		Ports: []current.ServicePort{
			{Protocol: "UDP", Port: 9376, TargetPort: util.NewIntOrStringFromString("p")},
			{Protocol: "UDP", Port: 8675, TargetPort: util.NewIntOrStringFromInt(309)},
		},
	}}
	out := roundTrip(t, runtime.Object(in)).(*current.Service)
	if out.Spec.Ports[0].Protocol != current.ProtocolUDP {
		t.Errorf("Expected protocol %s, got %s", current.ProtocolUDP, out.Spec.Ports[0].Protocol)
	}
	if out.Spec.Ports[0].TargetPort != util.NewIntOrStringFromString("p") {
		t.Errorf("Expected port %d, got %s", in.Spec.Ports[0].Port, out.Spec.Ports[0].TargetPort)
	}
	if out.Spec.Ports[1].Protocol != current.ProtocolUDP {
		t.Errorf("Expected protocol %s, got %s", current.ProtocolUDP, out.Spec.Ports[1].Protocol)
	}
	if out.Spec.Ports[1].TargetPort != util.NewIntOrStringFromInt(309) {
		t.Errorf("Expected port %d, got %s", in.Spec.Ports[1].Port, out.Spec.Ports[1].TargetPort)
	}

	// Defaulted.
	in = &current.Service{Spec: current.ServiceSpec{
		Ports: []current.ServicePort{
			{Protocol: "", Port: 9376, TargetPort: util.NewIntOrStringFromString("")},
			{Protocol: "", Port: 8675, TargetPort: util.NewIntOrStringFromInt(0)},
		},
	}}
	out = roundTrip(t, runtime.Object(in)).(*current.Service)
	if out.Spec.Ports[0].Protocol != current.ProtocolTCP {
		t.Errorf("Expected protocol %s, got %s", current.ProtocolTCP, out.Spec.Ports[0].Protocol)
	}
	if out.Spec.Ports[0].TargetPort != util.NewIntOrStringFromInt(in.Spec.Ports[0].Port) {
		t.Errorf("Expected port %d, got %d", in.Spec.Ports[0].Port, out.Spec.Ports[0].TargetPort)
	}
	if out.Spec.Ports[1].Protocol != current.ProtocolTCP {
		t.Errorf("Expected protocol %s, got %s", current.ProtocolTCP, out.Spec.Ports[1].Protocol)
	}
	if out.Spec.Ports[1].TargetPort != util.NewIntOrStringFromInt(in.Spec.Ports[1].Port) {
		t.Errorf("Expected port %d, got %d", in.Spec.Ports[1].Port, out.Spec.Ports[1].TargetPort)
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

func TestSetDefaultPodSpecHostNetwork(t *testing.T) {
	portNum := 8080
	s := current.PodSpec{}
	s.HostNetwork = true
	s.Containers = []current.Container{
		{
			Ports: []current.ContainerPort{
				{
					ContainerPort: portNum,
				},
			},
		},
	}
	pod := &current.Pod{
		Spec: s,
	}
	obj2 := roundTrip(t, runtime.Object(pod))
	pod2 := obj2.(*current.Pod)
	s2 := pod2.Spec

	hostPortNum := s2.Containers[0].Ports[0].HostPort
	if hostPortNum != portNum {
		t.Errorf("Expected container port to be defaulted, was made %d instead of %d", hostPortNum, portNum)
	}
}

func TestSetDefaultNodeExternalID(t *testing.T) {
	name := "node0"
	n := &current.Node{}
	n.Name = name
	obj2 := roundTrip(t, runtime.Object(n))
	n2 := obj2.(*current.Node)
	if n2.Spec.ExternalID != name {
		t.Errorf("Expected default External ID: %s, got: %s", name, n2.Spec.ExternalID)
	}
}

func TestSetDefaultObjectFieldSelectorAPIVersion(t *testing.T) {
	s := current.PodSpec{
		Containers: []current.Container{
			{
				Env: []current.EnvVar{
					{
						ValueFrom: &current.EnvVarSource{
							FieldRef: &current.ObjectFieldSelector{},
						},
					},
				},
			},
		},
	}
	pod := &current.Pod{
		Spec: s,
	}
	obj2 := roundTrip(t, runtime.Object(pod))
	pod2 := obj2.(*current.Pod)
	s2 := pod2.Spec

	apiVersion := s2.Containers[0].Env[0].ValueFrom.FieldRef.APIVersion
	if apiVersion != "v1" {
		t.Errorf("Expected default APIVersion v1, got: %v", apiVersion)
	}
}

func TestSetDefaultSecurityContext(t *testing.T) {
	priv := false
	privTrue := true
	testCases := map[string]struct {
		c current.Container
	}{
		"downward defaulting caps": {
			c: current.Container{
				Privileged: false,
				Capabilities: current.Capabilities{
					Add:  []current.CapabilityType{"foo"},
					Drop: []current.CapabilityType{"bar"},
				},
				SecurityContext: &current.SecurityContext{
					Privileged: &priv,
				},
			},
		},
		"downward defaulting priv": {
			c: current.Container{
				Privileged: false,
				Capabilities: current.Capabilities{
					Add:  []current.CapabilityType{"foo"},
					Drop: []current.CapabilityType{"bar"},
				},
				SecurityContext: &current.SecurityContext{
					Capabilities: &current.Capabilities{
						Add:  []current.CapabilityType{"foo"},
						Drop: []current.CapabilityType{"bar"},
					},
				},
			},
		},
		"upward defaulting caps": {
			c: current.Container{
				Privileged: false,
				SecurityContext: &current.SecurityContext{
					Privileged: &priv,
					Capabilities: &current.Capabilities{
						Add:  []current.CapabilityType{"biz"},
						Drop: []current.CapabilityType{"baz"},
					},
				},
			},
		},
		"upward defaulting priv": {
			c: current.Container{
				Capabilities: current.Capabilities{
					Add:  []current.CapabilityType{"foo"},
					Drop: []current.CapabilityType{"bar"},
				},
				SecurityContext: &current.SecurityContext{
					Privileged: &privTrue,
					Capabilities: &current.Capabilities{
						Add:  []current.CapabilityType{"foo"},
						Drop: []current.CapabilityType{"bar"},
					},
				},
			},
		},
	}

	pod := &current.Pod{
		Spec: current.PodSpec{},
	}

	for k, v := range testCases {
		pod.Spec.Containers = []current.Container{v.c}
		obj := roundTrip(t, runtime.Object(pod))
		defaultedPod := obj.(*current.Pod)
		c := defaultedPod.Spec.Containers[0]
		if isEqual, issues := areSecurityContextAndContainerEqual(&c); !isEqual {
			t.Errorf("test case %s expected the security context to have the same values as the container but found %#v", k, issues)
		}
	}
}

func areSecurityContextAndContainerEqual(c *current.Container) (bool, []string) {
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
