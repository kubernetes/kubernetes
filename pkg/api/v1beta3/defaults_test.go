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

package v1beta3_test

import (
	"reflect"
	"testing"

	"k8s.io/kubernetes/pkg/api"
	versioned "k8s.io/kubernetes/pkg/api/v1beta3"
	"k8s.io/kubernetes/pkg/apimachinery/registered"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/util/intstr"
)

func roundTrip(t *testing.T, obj runtime.Object) runtime.Object {
	data, err := runtime.Encode(v1beta3Codec, obj)
	if err != nil {
		t.Errorf("%v\n %#v", err, obj)
		return nil
	}
	obj2, err := runtime.Decode(v1beta3Codec, data)
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
	if !registered.IsAllowedVersion(versioned.SchemeGroupVersion) {
		return
	}

	tests := []struct {
		rc             *versioned.ReplicationController
		expectLabels   bool
		expectSelector bool
	}{
		{
			rc: &versioned.ReplicationController{
				Spec: versioned.ReplicationControllerSpec{
					Template: &versioned.PodTemplateSpec{
						ObjectMeta: versioned.ObjectMeta{
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
			rc: &versioned.ReplicationController{
				ObjectMeta: versioned.ObjectMeta{
					Labels: map[string]string{
						"bar": "foo",
					},
				},
				Spec: versioned.ReplicationControllerSpec{
					Template: &versioned.PodTemplateSpec{
						ObjectMeta: versioned.ObjectMeta{
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
			rc: &versioned.ReplicationController{
				ObjectMeta: versioned.ObjectMeta{
					Labels: map[string]string{
						"bar": "foo",
					},
				},
				Spec: versioned.ReplicationControllerSpec{
					Selector: map[string]string{
						"some": "other",
					},
					Template: &versioned.PodTemplateSpec{
						ObjectMeta: versioned.ObjectMeta{
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
			rc: &versioned.ReplicationController{
				Spec: versioned.ReplicationControllerSpec{
					Selector: map[string]string{
						"some": "other",
					},
					Template: &versioned.PodTemplateSpec{
						ObjectMeta: versioned.ObjectMeta{
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
		rc2, ok := obj2.(*versioned.ReplicationController)
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
	if !registered.IsAllowedVersion(versioned.SchemeGroupVersion) {
		return
	}

	svc := &versioned.Service{}
	obj2 := roundTrip(t, runtime.Object(svc))
	svc2 := obj2.(*versioned.Service)
	if svc2.Spec.SessionAffinity != versioned.ServiceAffinityNone {
		t.Errorf("Expected default session affinity type:%s, got: %s", versioned.ServiceAffinityNone, svc2.Spec.SessionAffinity)
	}
	if svc2.Spec.Type != versioned.ServiceTypeClusterIP {
		t.Errorf("Expected default type:%s, got: %s", versioned.ServiceTypeClusterIP, svc2.Spec.Type)
	}
}

func TestSetDefaultServiceWithLoadbalancer(t *testing.T) {
	if !registered.IsAllowedVersion(versioned.SchemeGroupVersion) {
		return
	}

	svc := &versioned.Service{}
	svc.Spec.CreateExternalLoadBalancer = true
	obj2 := roundTrip(t, runtime.Object(svc))
	svc2 := obj2.(*versioned.Service)
	if svc2.Spec.Type != versioned.ServiceTypeLoadBalancer {
		t.Errorf("Expected default type:%s, got: %s", versioned.ServiceTypeLoadBalancer, svc2.Spec.Type)
	}
}

func TestSetDefaultSecret(t *testing.T) {
	if !registered.IsAllowedVersion(versioned.SchemeGroupVersion) {
		return
	}

	s := &versioned.Secret{}
	obj2 := roundTrip(t, runtime.Object(s))
	s2 := obj2.(*versioned.Secret)

	if s2.Type != versioned.SecretTypeOpaque {
		t.Errorf("Expected secret type %v, got %v", versioned.SecretTypeOpaque, s2.Type)
	}
}

func TestSetDefaultPersistentVolume(t *testing.T) {
	if !registered.IsAllowedVersion(versioned.SchemeGroupVersion) {
		return
	}

	pv := &versioned.PersistentVolume{}
	obj2 := roundTrip(t, runtime.Object(pv))
	pv2 := obj2.(*versioned.PersistentVolume)

	if pv2.Status.Phase != versioned.VolumePending {
		t.Errorf("Expected volume phase %v, got %v", versioned.VolumePending, pv2.Status.Phase)
	}
	if pv2.Spec.PersistentVolumeReclaimPolicy != versioned.PersistentVolumeReclaimRetain {
		t.Errorf("Expected pv reclaim policy %v, got %v", versioned.PersistentVolumeReclaimRetain, pv2.Spec.PersistentVolumeReclaimPolicy)
	}
}

func TestSetDefaultPersistentVolumeClaim(t *testing.T) {
	if !registered.IsAllowedVersion(versioned.SchemeGroupVersion) {
		return
	}

	pvc := &versioned.PersistentVolumeClaim{}
	obj2 := roundTrip(t, runtime.Object(pvc))
	pvc2 := obj2.(*versioned.PersistentVolumeClaim)

	if pvc2.Status.Phase != versioned.ClaimPending {
		t.Errorf("Expected claim phase %v, got %v", versioned.ClaimPending, pvc2.Status.Phase)
	}
}

func TestSetDefaulEndpointsProtocol(t *testing.T) {
	if !registered.IsAllowedVersion(versioned.SchemeGroupVersion) {
		return
	}

	in := &versioned.Endpoints{Subsets: []versioned.EndpointSubset{
		{Ports: []versioned.EndpointPort{{}, {Protocol: "UDP"}, {}}},
	}}
	obj := roundTrip(t, runtime.Object(in))
	out := obj.(*versioned.Endpoints)

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

func TestSetDefaulServiceTargetPort(t *testing.T) {
	if !registered.IsAllowedVersion(versioned.SchemeGroupVersion) {
		return
	}

	in := &versioned.Service{Spec: versioned.ServiceSpec{Ports: []versioned.ServicePort{{Port: 1234}}}}
	obj := roundTrip(t, runtime.Object(in))
	out := obj.(*versioned.Service)
	if out.Spec.Ports[0].TargetPort != intstr.FromInt(1234) {
		t.Errorf("Expected TargetPort to be defaulted, got %s", out.Spec.Ports[0].TargetPort)
	}

	in = &versioned.Service{Spec: versioned.ServiceSpec{Ports: []versioned.ServicePort{{Port: 1234, TargetPort: intstr.FromInt(5678)}}}}
	obj = roundTrip(t, runtime.Object(in))
	out = obj.(*versioned.Service)
	if out.Spec.Ports[0].TargetPort != intstr.FromInt(5678) {
		t.Errorf("Expected TargetPort to be unchanged, got %s", out.Spec.Ports[0].TargetPort)
	}
}

func TestSetDefaultServicePort(t *testing.T) {
	if !registered.IsAllowedVersion(versioned.SchemeGroupVersion) {
		return
	}

	// Unchanged if set.
	in := &versioned.Service{Spec: versioned.ServiceSpec{
		Ports: []versioned.ServicePort{
			{Protocol: "UDP", Port: 9376, TargetPort: intstr.FromString("p")},
			{Protocol: "UDP", Port: 8675, TargetPort: intstr.FromInt(309)},
		},
	}}
	out := roundTrip(t, runtime.Object(in)).(*versioned.Service)
	if out.Spec.Ports[0].Protocol != versioned.ProtocolUDP {
		t.Errorf("Expected protocol %s, got %s", versioned.ProtocolUDP, out.Spec.Ports[0].Protocol)
	}
	if out.Spec.Ports[0].TargetPort != intstr.FromString("p") {
		t.Errorf("Expected port %d, got %s", in.Spec.Ports[0].Port, out.Spec.Ports[0].TargetPort)
	}
	if out.Spec.Ports[1].Protocol != versioned.ProtocolUDP {
		t.Errorf("Expected protocol %s, got %s", versioned.ProtocolUDP, out.Spec.Ports[1].Protocol)
	}
	if out.Spec.Ports[1].TargetPort != intstr.FromInt(309) {
		t.Errorf("Expected port %d, got %s", in.Spec.Ports[1].Port, out.Spec.Ports[1].TargetPort)
	}

	// Defaulted.
	in = &versioned.Service{Spec: versioned.ServiceSpec{
		Ports: []versioned.ServicePort{
			{Protocol: "", Port: 9376, TargetPort: intstr.FromString("")},
			{Protocol: "", Port: 8675, TargetPort: intstr.FromInt(0)},
		},
	}}
	out = roundTrip(t, runtime.Object(in)).(*versioned.Service)
	if out.Spec.Ports[0].Protocol != versioned.ProtocolTCP {
		t.Errorf("Expected protocol %s, got %s", versioned.ProtocolTCP, out.Spec.Ports[0].Protocol)
	}
	if out.Spec.Ports[0].TargetPort != intstr.FromInt(in.Spec.Ports[0].Port) {
		t.Errorf("Expected port %d, got %d", in.Spec.Ports[0].Port, out.Spec.Ports[0].TargetPort)
	}
	if out.Spec.Ports[1].Protocol != versioned.ProtocolTCP {
		t.Errorf("Expected protocol %s, got %s", versioned.ProtocolTCP, out.Spec.Ports[1].Protocol)
	}
	if out.Spec.Ports[1].TargetPort != intstr.FromInt(in.Spec.Ports[1].Port) {
		t.Errorf("Expected port %d, got %d", in.Spec.Ports[1].Port, out.Spec.Ports[1].TargetPort)
	}
}

func TestSetDefaultNamespace(t *testing.T) {
	if !registered.IsAllowedVersion(versioned.SchemeGroupVersion) {
		return
	}

	s := &versioned.Namespace{}
	obj2 := roundTrip(t, runtime.Object(s))
	s2 := obj2.(*versioned.Namespace)

	if s2.Status.Phase != versioned.NamespaceActive {
		t.Errorf("Expected phase %v, got %v", versioned.NamespaceActive, s2.Status.Phase)
	}
}

func TestSetDefaultPodSpecHostNetwork(t *testing.T) {
	if !registered.IsAllowedVersion(versioned.SchemeGroupVersion) {
		return
	}

	portNum := 8080
	s := versioned.PodSpec{}
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
	pod := &versioned.Pod{
		Spec: s,
	}
	obj2 := roundTrip(t, runtime.Object(pod))
	pod2 := obj2.(*versioned.Pod)
	s2 := pod2.Spec

	hostPortNum := s2.Containers[0].Ports[0].HostPort
	if hostPortNum != portNum {
		t.Errorf("Expected container port to be defaulted, was made %d instead of %d", hostPortNum, portNum)
	}
}

func TestSetDefaultNodeExternalID(t *testing.T) {
	if !registered.IsAllowedVersion(versioned.SchemeGroupVersion) {
		return
	}

	name := "node0"
	n := &versioned.Node{}
	n.Name = name
	obj2 := roundTrip(t, runtime.Object(n))
	n2 := obj2.(*versioned.Node)
	if n2.Spec.ExternalID != name {
		t.Errorf("Expected default External ID: %s, got: %s", name, n2.Spec.ExternalID)
	}
	if n2.Spec.ProviderID != "" {
		t.Errorf("Expected empty default Cloud Provider ID, got: %s", n2.Spec.ProviderID)
	}
}

func TestSetDefaultObjectFieldSelectorAPIVersion(t *testing.T) {
	if !registered.IsAllowedVersion(versioned.SchemeGroupVersion) {
		return
	}

	s := versioned.PodSpec{
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
	pod := &versioned.Pod{
		Spec: s,
	}
	obj2 := roundTrip(t, runtime.Object(pod))
	pod2 := obj2.(*versioned.Pod)
	s2 := pod2.Spec

	apiVersion := s2.Containers[0].Env[0].ValueFrom.FieldRef.APIVersion
	if apiVersion != "v1beta3" {
		t.Errorf("Expected default APIVersion v1beta3, got: %v", apiVersion)
	}
}

func TestSetDefaultSecurityContext(t *testing.T) {
	if !registered.IsAllowedVersion(versioned.SchemeGroupVersion) {
		return
	}

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
		Spec: versioned.PodSpec{},
	}

	for k, v := range testCases {
		pod.Spec.Containers = []versioned.Container{v.c}
		obj := roundTrip(t, runtime.Object(pod))
		defaultedPod := obj.(*versioned.Pod)
		c := defaultedPod.Spec.Containers[0]
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

func TestDefaultSecurityContextConstraints(t *testing.T) {
	if !registered.IsAllowedVersion(versioned.SchemeGroupVersion) {
		return
	}

	tests := map[string]struct {
		scc              *versioned.SecurityContextConstraints
		expectedFSGroup  versioned.FSGroupStrategyType
		expectedSupGroup versioned.SupplementalGroupsStrategyType
	}{
		"shouldn't default": {
			scc: &versioned.SecurityContextConstraints{
				FSGroup: versioned.FSGroupStrategyOptions{
					Type: versioned.FSGroupStrategyMustRunAs,
				},
				SupplementalGroups: versioned.SupplementalGroupsStrategyOptions{
					Type: versioned.SupplementalGroupsStrategyMustRunAs,
				},
			},
			expectedFSGroup:  versioned.FSGroupStrategyMustRunAs,
			expectedSupGroup: versioned.SupplementalGroupsStrategyMustRunAs,
		},
		"default fsgroup runAsAny": {
			scc: &versioned.SecurityContextConstraints{
				RunAsUser: versioned.RunAsUserStrategyOptions{
					Type: versioned.RunAsUserStrategyRunAsAny,
				},
				SupplementalGroups: versioned.SupplementalGroupsStrategyOptions{
					Type: versioned.SupplementalGroupsStrategyMustRunAs,
				},
			},
			expectedFSGroup:  versioned.FSGroupStrategyRunAsAny,
			expectedSupGroup: versioned.SupplementalGroupsStrategyMustRunAs,
		},
		"default sup group runAsAny": {
			scc: &versioned.SecurityContextConstraints{
				RunAsUser: versioned.RunAsUserStrategyOptions{
					Type: versioned.RunAsUserStrategyRunAsAny,
				},
				FSGroup: versioned.FSGroupStrategyOptions{
					Type: versioned.FSGroupStrategyMustRunAs,
				},
			},
			expectedFSGroup:  versioned.FSGroupStrategyMustRunAs,
			expectedSupGroup: versioned.SupplementalGroupsStrategyRunAsAny,
		},
		"default fsgroup runAsAny with mustRunAs UID strat": {
			scc: &versioned.SecurityContextConstraints{
				RunAsUser: versioned.RunAsUserStrategyOptions{
					Type: versioned.RunAsUserStrategyMustRunAsRange,
				},
				SupplementalGroups: versioned.SupplementalGroupsStrategyOptions{
					Type: versioned.SupplementalGroupsStrategyMustRunAs,
				},
			},
			expectedFSGroup:  versioned.FSGroupStrategyRunAsAny,
			expectedSupGroup: versioned.SupplementalGroupsStrategyMustRunAs,
		},
		"default sup group runAsAny with mustRunAs UID strat": {
			scc: &versioned.SecurityContextConstraints{
				RunAsUser: versioned.RunAsUserStrategyOptions{
					Type: versioned.RunAsUserStrategyMustRunAsRange,
				},
				FSGroup: versioned.FSGroupStrategyOptions{
					Type: versioned.FSGroupStrategyMustRunAs,
				},
			},
			expectedFSGroup:  versioned.FSGroupStrategyMustRunAs,
			expectedSupGroup: versioned.SupplementalGroupsStrategyRunAsAny,
		},
	}
	for k, v := range tests {
		output := roundTrip(t, runtime.Object(v.scc))
		scc := output.(*versioned.SecurityContextConstraints)

		if scc.FSGroup.Type != v.expectedFSGroup {
			t.Errorf("%s has invalid fsgroup.  Expected: %v got: %v", k, v.expectedFSGroup, scc.FSGroup.Type)
		}
		if scc.SupplementalGroups.Type != v.expectedSupGroup {
			t.Errorf("%s has invalid supplemental group.  Expected: %v got: %v", k, v.expectedSupGroup, scc.SupplementalGroups.Type)
		}
	}
}
