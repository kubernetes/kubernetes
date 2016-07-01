/*
Copyright 2015 The Kubernetes Authors.

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

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/resource"
	versioned "k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/util/intstr"
)

func roundTrip(t *testing.T, obj runtime.Object) runtime.Object {
	codec := api.Codecs.LegacyCodec(versioned.SchemeGroupVersion)
	data, err := runtime.Encode(codec, obj)
	if err != nil {
		t.Errorf("%v\n %#v", err, obj)
		return nil
	}
	obj2, err := runtime.Decode(codec, data)
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

func newInt(val int32) *int32 {
	p := new(int32)
	*p = val
	return p
}

func TestSetDefaultReplicationControllerReplicas(t *testing.T) {
	tests := []struct {
		rc             versioned.ReplicationController
		expectReplicas int32
	}{
		{
			rc: versioned.ReplicationController{
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
			expectReplicas: 1,
		},
		{
			rc: versioned.ReplicationController{
				Spec: versioned.ReplicationControllerSpec{
					Replicas: newInt(0),
					Template: &versioned.PodTemplateSpec{
						ObjectMeta: versioned.ObjectMeta{
							Labels: map[string]string{
								"foo": "bar",
							},
						},
					},
				},
			},
			expectReplicas: 0,
		},
		{
			rc: versioned.ReplicationController{
				Spec: versioned.ReplicationControllerSpec{
					Replicas: newInt(3),
					Template: &versioned.PodTemplateSpec{
						ObjectMeta: versioned.ObjectMeta{
							Labels: map[string]string{
								"foo": "bar",
							},
						},
					},
				},
			},
			expectReplicas: 3,
		},
	}

	for _, test := range tests {
		rc := &test.rc
		obj2 := roundTrip(t, runtime.Object(rc))
		rc2, ok := obj2.(*versioned.ReplicationController)
		if !ok {
			t.Errorf("unexpected object: %v", rc2)
			t.FailNow()
		}
		if rc2.Spec.Replicas == nil {
			t.Errorf("unexpected nil Replicas")
		} else if test.expectReplicas != *rc2.Spec.Replicas {
			t.Errorf("expected: %d replicas, got: %d", test.expectReplicas, *rc2.Spec.Replicas)
		}
	}
}

func TestSetDefaultService(t *testing.T) {
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

func TestSetDefaultSecret(t *testing.T) {
	s := &versioned.Secret{}
	obj2 := roundTrip(t, runtime.Object(s))
	s2 := obj2.(*versioned.Secret)

	if s2.Type != versioned.SecretTypeOpaque {
		t.Errorf("Expected secret type %v, got %v", versioned.SecretTypeOpaque, s2.Type)
	}
}

func TestSetDefaultPersistentVolume(t *testing.T) {
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
	pvc := &versioned.PersistentVolumeClaim{}
	obj2 := roundTrip(t, runtime.Object(pvc))
	pvc2 := obj2.(*versioned.PersistentVolumeClaim)

	if pvc2.Status.Phase != versioned.ClaimPending {
		t.Errorf("Expected claim phase %v, got %v", versioned.ClaimPending, pvc2.Status.Phase)
	}
}

func TestSetDefaulEndpointsProtocol(t *testing.T) {
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
	in := &versioned.Service{Spec: versioned.ServiceSpec{Ports: []versioned.ServicePort{{Port: 1234}}}}
	obj := roundTrip(t, runtime.Object(in))
	out := obj.(*versioned.Service)
	if out.Spec.Ports[0].TargetPort != intstr.FromInt(1234) {
		t.Errorf("Expected TargetPort to be defaulted, got %v", out.Spec.Ports[0].TargetPort)
	}

	in = &versioned.Service{Spec: versioned.ServiceSpec{Ports: []versioned.ServicePort{{Port: 1234, TargetPort: intstr.FromInt(5678)}}}}
	obj = roundTrip(t, runtime.Object(in))
	out = obj.(*versioned.Service)
	if out.Spec.Ports[0].TargetPort != intstr.FromInt(5678) {
		t.Errorf("Expected TargetPort to be unchanged, got %v", out.Spec.Ports[0].TargetPort)
	}
}

func TestSetDefaultServicePort(t *testing.T) {
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
		t.Errorf("Expected port %v, got %v", in.Spec.Ports[0].Port, out.Spec.Ports[0].TargetPort)
	}
	if out.Spec.Ports[1].Protocol != versioned.ProtocolUDP {
		t.Errorf("Expected protocol %s, got %s", versioned.ProtocolUDP, out.Spec.Ports[1].Protocol)
	}
	if out.Spec.Ports[1].TargetPort != intstr.FromInt(309) {
		t.Errorf("Expected port %v, got %v", in.Spec.Ports[1].Port, out.Spec.Ports[1].TargetPort)
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
	if out.Spec.Ports[0].TargetPort != intstr.FromInt(int(in.Spec.Ports[0].Port)) {
		t.Errorf("Expected port %v, got %v", in.Spec.Ports[0].Port, out.Spec.Ports[0].TargetPort)
	}
	if out.Spec.Ports[1].Protocol != versioned.ProtocolTCP {
		t.Errorf("Expected protocol %s, got %s", versioned.ProtocolTCP, out.Spec.Ports[1].Protocol)
	}
	if out.Spec.Ports[1].TargetPort != intstr.FromInt(int(in.Spec.Ports[1].Port)) {
		t.Errorf("Expected port %v, got %v", in.Spec.Ports[1].Port, out.Spec.Ports[1].TargetPort)
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

func TestSetDefaultPodSpecHostNetwork(t *testing.T) {
	portNum := int32(8080)
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

func TestSetDefaultNodeStatusAllocatable(t *testing.T) {
	capacity := versioned.ResourceList{
		versioned.ResourceCPU:    resource.MustParse("1000m"),
		versioned.ResourceMemory: resource.MustParse("10G"),
	}
	allocatable := versioned.ResourceList{
		versioned.ResourceCPU:    resource.MustParse("500m"),
		versioned.ResourceMemory: resource.MustParse("5G"),
	}
	tests := []struct {
		capacity            versioned.ResourceList
		allocatable         versioned.ResourceList
		expectedAllocatable versioned.ResourceList
	}{{ // Everything set, no defaulting.
		capacity:            capacity,
		allocatable:         allocatable,
		expectedAllocatable: allocatable,
	}, { // Allocatable set, no defaulting.
		capacity:            nil,
		allocatable:         allocatable,
		expectedAllocatable: allocatable,
	}, { // Capacity set, allocatable defaults to capacity.
		capacity:            capacity,
		allocatable:         nil,
		expectedAllocatable: capacity,
	}, { // Nothing set, allocatable "defaults" to capacity.
		capacity:            nil,
		allocatable:         nil,
		expectedAllocatable: nil,
	}}

	copyResourceList := func(rl versioned.ResourceList) versioned.ResourceList {
		if rl == nil {
			return nil
		}
		copy := make(versioned.ResourceList, len(rl))
		for k, v := range rl {
			copy[k] = *v.Copy()
		}
		return copy
	}

	resourceListsEqual := func(a versioned.ResourceList, b versioned.ResourceList) bool {
		if len(a) != len(b) {
			return false
		}
		for k, v := range a {
			vb, found := b[k]
			if !found {
				return false
			}
			if v.Cmp(vb) != 0 {
				return false
			}
		}
		return true
	}

	for i, testcase := range tests {
		node := versioned.Node{
			Status: versioned.NodeStatus{
				Capacity:    copyResourceList(testcase.capacity),
				Allocatable: copyResourceList(testcase.allocatable),
			},
		}
		node2 := roundTrip(t, runtime.Object(&node)).(*versioned.Node)
		actual := node2.Status.Allocatable
		expected := testcase.expectedAllocatable
		if !resourceListsEqual(expected, actual) {
			t.Errorf("[%d] Expected NodeStatus.Allocatable: %+v; Got: %+v", i, expected, actual)
		}
	}
}

func TestSetDefaultObjectFieldSelectorAPIVersion(t *testing.T) {
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
	if apiVersion != "v1" {
		t.Errorf("Expected default APIVersion v1, got: %v", apiVersion)
	}
}

func TestSetDefaultRequestsPod(t *testing.T) {
	// verify we default if limits are specified (and that request=0 is preserved)
	s := versioned.PodSpec{}
	s.Containers = []versioned.Container{
		{
			Resources: versioned.ResourceRequirements{
				Requests: versioned.ResourceList{
					versioned.ResourceMemory: resource.MustParse("0"),
				},
				Limits: versioned.ResourceList{
					versioned.ResourceCPU:    resource.MustParse("100m"),
					versioned.ResourceMemory: resource.MustParse("1Gi"),
				},
			},
		},
	}
	pod := &versioned.Pod{
		Spec: s,
	}
	output := roundTrip(t, runtime.Object(pod))
	pod2 := output.(*versioned.Pod)
	defaultRequest := pod2.Spec.Containers[0].Resources.Requests
	if requestValue := defaultRequest[versioned.ResourceCPU]; requestValue.String() != "100m" {
		t.Errorf("Expected request cpu: %s, got: %s", "100m", requestValue.String())
	}
	if requestValue := defaultRequest[versioned.ResourceMemory]; requestValue.String() != "0" {
		t.Errorf("Expected request memory: %s, got: %s", "0", requestValue.String())
	}

	// verify we do nothing if no limits are specified
	s = versioned.PodSpec{}
	s.Containers = []versioned.Container{{}}
	pod = &versioned.Pod{
		Spec: s,
	}
	output = roundTrip(t, runtime.Object(pod))
	pod2 = output.(*versioned.Pod)
	defaultRequest = pod2.Spec.Containers[0].Resources.Requests
	if requestValue := defaultRequest[versioned.ResourceCPU]; requestValue.String() != "0" {
		t.Errorf("Expected 0 request value, got: %s", requestValue.String())
	}
}

func TestDefaultRequestIsNotSetForReplicationController(t *testing.T) {
	s := versioned.PodSpec{}
	s.Containers = []versioned.Container{
		{
			Resources: versioned.ResourceRequirements{
				Limits: versioned.ResourceList{
					versioned.ResourceCPU: resource.MustParse("100m"),
				},
			},
		},
	}
	rc := &versioned.ReplicationController{
		Spec: versioned.ReplicationControllerSpec{
			Replicas: newInt(3),
			Template: &versioned.PodTemplateSpec{
				ObjectMeta: versioned.ObjectMeta{
					Labels: map[string]string{
						"foo": "bar",
					},
				},
				Spec: s,
			},
		},
	}
	output := roundTrip(t, runtime.Object(rc))
	rc2 := output.(*versioned.ReplicationController)
	defaultRequest := rc2.Spec.Template.Spec.Containers[0].Resources.Requests
	requestValue := defaultRequest[versioned.ResourceCPU]
	if requestValue.String() != "0" {
		t.Errorf("Expected 0 request value, got: %s", requestValue.String())
	}
}

func TestSetDefaultLimitRangeItem(t *testing.T) {
	limitRange := &versioned.LimitRange{
		ObjectMeta: versioned.ObjectMeta{
			Name: "test-defaults",
		},
		Spec: versioned.LimitRangeSpec{
			Limits: []versioned.LimitRangeItem{{
				Type: versioned.LimitTypeContainer,
				Max: versioned.ResourceList{
					versioned.ResourceCPU: resource.MustParse("100m"),
				},
				Min: versioned.ResourceList{
					versioned.ResourceMemory: resource.MustParse("100Mi"),
				},
				Default:        versioned.ResourceList{},
				DefaultRequest: versioned.ResourceList{},
			}},
		},
	}

	output := roundTrip(t, runtime.Object(limitRange))
	limitRange2 := output.(*versioned.LimitRange)
	defaultLimit := limitRange2.Spec.Limits[0].Default
	defaultRequest := limitRange2.Spec.Limits[0].DefaultRequest

	// verify that default cpu was set to the max
	defaultValue := defaultLimit[versioned.ResourceCPU]
	if defaultValue.String() != "100m" {
		t.Errorf("Expected default cpu: %s, got: %s", "100m", defaultValue.String())
	}
	// verify that default request was set to the limit
	requestValue := defaultRequest[versioned.ResourceCPU]
	if requestValue.String() != "100m" {
		t.Errorf("Expected request cpu: %s, got: %s", "100m", requestValue.String())
	}
	// verify that if a min is provided, it will be the default if no limit is specified
	requestMinValue := defaultRequest[versioned.ResourceMemory]
	if requestMinValue.String() != "100Mi" {
		t.Errorf("Expected request memory: %s, got: %s", "100Mi", requestMinValue.String())
	}
}

func TestSetDefaultProbe(t *testing.T) {
	originalProbe := versioned.Probe{}
	expectedProbe := versioned.Probe{
		InitialDelaySeconds: 0,
		TimeoutSeconds:      1,
		PeriodSeconds:       10,
		SuccessThreshold:    1,
		FailureThreshold:    3,
	}

	pod := &versioned.Pod{
		Spec: versioned.PodSpec{
			Containers: []versioned.Container{{LivenessProbe: &originalProbe}},
		},
	}

	output := roundTrip(t, runtime.Object(pod)).(*versioned.Pod)
	actualProbe := *output.Spec.Containers[0].LivenessProbe
	if actualProbe != expectedProbe {
		t.Errorf("Expected probe: %+v\ngot: %+v\n", expectedProbe, actualProbe)
	}
}
