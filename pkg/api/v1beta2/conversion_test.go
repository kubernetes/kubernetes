/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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
	"encoding/json"
	"reflect"
	"testing"

	newer "github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/resource"
	current "github.com/GoogleCloudPlatform/kubernetes/pkg/api/v1beta2"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
)

func TestServiceEmptySelector(t *testing.T) {
	// Nil map should be preserved
	svc := &current.Service{Selector: nil}
	data, err := newer.Scheme.EncodeToVersion(svc, "v1beta2")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	obj, err := newer.Scheme.Decode(data)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	selector := obj.(*newer.Service).Spec.Selector
	if selector != nil {
		t.Errorf("unexpected selector: %#v", obj)
	}

	// Empty map should be preserved
	svc2 := &current.Service{Selector: map[string]string{}}
	data, err = newer.Scheme.EncodeToVersion(svc2, "v1beta2")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	obj, err = newer.Scheme.Decode(data)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	selector = obj.(*newer.Service).Spec.Selector
	if selector == nil || len(selector) != 0 {
		t.Errorf("unexpected selector: %#v", obj)
	}
}

func TestServicePorts(t *testing.T) {
	testCases := []struct {
		given     current.Service
		expected  newer.Service
		roundtrip current.Service
	}{
		{
			given: current.Service{
				TypeMeta: current.TypeMeta{
					ID: "legacy-with-defaults",
				},
				Port:     111,
				Protocol: current.ProtocolTCP,
			},
			expected: newer.Service{
				Spec: newer.ServiceSpec{Ports: []newer.ServicePort{{
					Port:     111,
					Protocol: newer.ProtocolTCP,
				}}},
			},
			roundtrip: current.Service{
				Ports: []current.ServicePort{{
					Port:     111,
					Protocol: current.ProtocolTCP,
				}},
			},
		},
		{
			given: current.Service{
				TypeMeta: current.TypeMeta{
					ID: "legacy-full",
				},
				PortName:      "p",
				Port:          111,
				Protocol:      current.ProtocolTCP,
				ContainerPort: util.NewIntOrStringFromString("p"),
			},
			expected: newer.Service{
				Spec: newer.ServiceSpec{Ports: []newer.ServicePort{{
					Name:       "p",
					Port:       111,
					Protocol:   newer.ProtocolTCP,
					TargetPort: util.NewIntOrStringFromString("p"),
				}}},
			},
			roundtrip: current.Service{
				Ports: []current.ServicePort{{
					Name:          "p",
					Port:          111,
					Protocol:      current.ProtocolTCP,
					ContainerPort: util.NewIntOrStringFromString("p"),
				}},
			},
		},
		{
			given: current.Service{
				TypeMeta: current.TypeMeta{
					ID: "both",
				},
				PortName:      "p",
				Port:          111,
				Protocol:      current.ProtocolTCP,
				ContainerPort: util.NewIntOrStringFromString("p"),
				Ports: []current.ServicePort{{
					Name:          "q",
					Port:          222,
					Protocol:      current.ProtocolUDP,
					ContainerPort: util.NewIntOrStringFromInt(93),
				}},
			},
			expected: newer.Service{
				Spec: newer.ServiceSpec{Ports: []newer.ServicePort{{
					Name:       "q",
					Port:       222,
					Protocol:   newer.ProtocolUDP,
					TargetPort: util.NewIntOrStringFromInt(93),
				}}},
			},
			roundtrip: current.Service{
				Ports: []current.ServicePort{{
					Name:          "q",
					Port:          222,
					Protocol:      current.ProtocolUDP,
					ContainerPort: util.NewIntOrStringFromInt(93),
				}},
			},
		},
		{
			given: current.Service{
				TypeMeta: current.TypeMeta{
					ID: "one",
				},
				Ports: []current.ServicePort{{
					Name:          "p",
					Port:          111,
					Protocol:      current.ProtocolUDP,
					ContainerPort: util.NewIntOrStringFromInt(93),
				}},
			},
			expected: newer.Service{
				Spec: newer.ServiceSpec{Ports: []newer.ServicePort{{
					Name:       "p",
					Port:       111,
					Protocol:   newer.ProtocolUDP,
					TargetPort: util.NewIntOrStringFromInt(93),
				}}},
			},
			roundtrip: current.Service{
				Ports: []current.ServicePort{{
					Name:          "p",
					Port:          111,
					Protocol:      current.ProtocolUDP,
					ContainerPort: util.NewIntOrStringFromInt(93),
				}},
			},
		},
		{
			given: current.Service{
				TypeMeta: current.TypeMeta{
					ID: "two",
				},
				Ports: []current.ServicePort{{
					Name:          "p",
					Port:          111,
					Protocol:      current.ProtocolUDP,
					ContainerPort: util.NewIntOrStringFromInt(93),
				}, {
					Name:          "q",
					Port:          222,
					Protocol:      current.ProtocolTCP,
					ContainerPort: util.NewIntOrStringFromInt(76),
				}},
			},
			expected: newer.Service{
				Spec: newer.ServiceSpec{Ports: []newer.ServicePort{{
					Name:       "p",
					Port:       111,
					Protocol:   newer.ProtocolUDP,
					TargetPort: util.NewIntOrStringFromInt(93),
				}, {
					Name:       "q",
					Port:       222,
					Protocol:   newer.ProtocolTCP,
					TargetPort: util.NewIntOrStringFromInt(76),
				}}},
			},
			roundtrip: current.Service{
				Ports: []current.ServicePort{{
					Name:          "p",
					Port:          111,
					Protocol:      current.ProtocolUDP,
					ContainerPort: util.NewIntOrStringFromInt(93),
				}, {
					Name:          "q",
					Port:          222,
					Protocol:      current.ProtocolTCP,
					ContainerPort: util.NewIntOrStringFromInt(76),
				}},
			},
		},
	}

	for i, tc := range testCases {
		// Convert versioned -> internal.
		got := newer.Service{}
		if err := newer.Scheme.Convert(&tc.given, &got); err != nil {
			t.Errorf("[Case: %d] Unexpected error: %v", i, err)
			continue
		}
		if !reflect.DeepEqual(got.Spec.Ports, tc.expected.Spec.Ports) {
			t.Errorf("[Case: %d] Expected %v, got %v", i, tc.expected.Spec.Ports, got.Spec.Ports)
		}

		// Convert internal -> versioned.
		got2 := current.Service{}
		if err := newer.Scheme.Convert(&got, &got2); err != nil {
			t.Errorf("[Case: %d] Unexpected error: %v", i, err)
			continue
		}
		if !reflect.DeepEqual(got2.Ports, tc.roundtrip.Ports) {
			t.Errorf("[Case: %d] Expected %v, got %v", i, tc.roundtrip.Ports, got2.Ports)
		}
	}
}

func TestNodeConversion(t *testing.T) {
	version, kind, err := newer.Scheme.ObjectVersionAndKind(&current.Minion{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if version != "v1beta2" || kind != "Minion" {
		t.Errorf("unexpected version and kind: %s %s", version, kind)
	}

	newer.Scheme.Log(t)
	obj, err := current.Codec.Decode([]byte(`{"kind":"Node","apiVersion":"v1beta2"}`))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if _, ok := obj.(*newer.Node); !ok {
		t.Errorf("unexpected type: %#v", obj)
	}

	obj, err = current.Codec.Decode([]byte(`{"kind":"NodeList","apiVersion":"v1beta2"}`))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if _, ok := obj.(*newer.NodeList); !ok {
		t.Errorf("unexpected type: %#v", obj)
	}

	obj = &newer.Node{}
	if err := current.Codec.DecodeInto([]byte(`{"kind":"Node","apiVersion":"v1beta2"}`), obj); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	obj = &newer.Node{}
	data, err := current.Codec.Encode(obj)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	m := map[string]interface{}{}
	if err := json.Unmarshal(data, &m); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if m["kind"] != "Minion" {
		t.Errorf("unexpected encoding: %s - %#v", m["kind"], string(data))
	}
}

func TestPullPolicyConversion(t *testing.T) {
	table := []struct {
		versioned current.PullPolicy
		internal  newer.PullPolicy
	}{
		{
			versioned: current.PullAlways,
			internal:  newer.PullAlways,
		}, {
			versioned: current.PullNever,
			internal:  newer.PullNever,
		}, {
			versioned: current.PullIfNotPresent,
			internal:  newer.PullIfNotPresent,
		}, {
			versioned: "",
			internal:  "",
		}, {
			versioned: "invalid value",
			internal:  "invalid value",
		},
	}
	for _, item := range table {
		var got newer.PullPolicy
		err := newer.Scheme.Convert(&item.versioned, &got)
		if err != nil {
			t.Errorf("Unexpected error: %v", err)
			continue
		}
		if e, a := item.internal, got; e != a {
			t.Errorf("Expected: %q, got %q", e, a)
		}
	}
	for _, item := range table {
		var got current.PullPolicy
		err := newer.Scheme.Convert(&item.internal, &got)
		if err != nil {
			t.Errorf("Unexpected error: %v", err)
			continue
		}
		if e, a := item.versioned, got; e != a {
			t.Errorf("Expected: %q, got %q", e, a)
		}
	}
}

func getResourceRequirements(cpu, memory resource.Quantity) current.ResourceRequirements {
	res := current.ResourceRequirements{}
	res.Limits = current.ResourceList{}
	if cpu.Value() > 0 {
		res.Limits[current.ResourceCPU] = util.NewIntOrStringFromInt(int(cpu.Value()))
	}
	if memory.Value() > 0 {
		res.Limits[current.ResourceMemory] = util.NewIntOrStringFromInt(int(memory.Value()))
	}

	return res
}

func TestContainerConversion(t *testing.T) {
	cpuLimit := resource.MustParse("10")
	memoryLimit := resource.MustParse("10M")
	null := resource.Quantity{}
	testCases := []current.Container{
		{
			Name:      "container",
			Resources: getResourceRequirements(cpuLimit, memoryLimit),
		},
		{
			Name:      "container",
			CPU:       int(cpuLimit.MilliValue()),
			Resources: getResourceRequirements(null, memoryLimit),
		},
		{
			Name:      "container",
			Memory:    memoryLimit.Value(),
			Resources: getResourceRequirements(cpuLimit, null),
		},
		{
			Name:   "container",
			CPU:    int(cpuLimit.MilliValue()),
			Memory: memoryLimit.Value(),
		},
		{
			Name:      "container",
			Memory:    memoryLimit.Value(),
			Resources: getResourceRequirements(cpuLimit, resource.MustParse("100M")),
		},
		{
			Name:      "container",
			CPU:       int(cpuLimit.MilliValue()),
			Resources: getResourceRequirements(resource.MustParse("500"), memoryLimit),
		},
	}

	for i, tc := range testCases {
		got := newer.Container{}
		if err := newer.Scheme.Convert(&tc, &got); err != nil {
			t.Errorf("[Case: %d] Unexpected error: %v", i, err)
			continue
		}
		if cpu := got.Resources.Limits.Cpu(); cpu.Value() != cpuLimit.Value() {
			t.Errorf("[Case: %d] Expected cpu: %v, got: %v", i, cpuLimit, *cpu)
		}
		if memory := got.Resources.Limits.Memory(); memory.Value() != memoryLimit.Value() {
			t.Errorf("[Case: %d] Expected memory: %v, got: %v", i, memoryLimit, *memory)
		}
	}
}

func TestEndpointsConversion(t *testing.T) {
	testCases := []struct {
		given    current.Endpoints
		expected newer.Endpoints
	}{
		{
			given: current.Endpoints{
				TypeMeta: current.TypeMeta{
					ID: "empty",
				},
				Protocol:  current.ProtocolTCP,
				Endpoints: []string{},
			},
			expected: newer.Endpoints{
				Subsets: []newer.EndpointSubset{},
			},
		},
		{
			given: current.Endpoints{
				TypeMeta: current.TypeMeta{
					ID: "one legacy",
				},
				Protocol:  current.ProtocolTCP,
				Endpoints: []string{"1.2.3.4:88"},
			},
			expected: newer.Endpoints{
				Subsets: []newer.EndpointSubset{{
					Ports:     []newer.EndpointPort{{Name: "", Port: 88, Protocol: newer.ProtocolTCP}},
					Addresses: []newer.EndpointAddress{{IP: "1.2.3.4"}},
				}},
			},
		},
		{
			given: current.Endpoints{
				TypeMeta: current.TypeMeta{
					ID: "several legacy",
				},
				Protocol:  current.ProtocolUDP,
				Endpoints: []string{"1.2.3.4:88", "1.2.3.4:89", "1.2.3.4:90"},
			},
			expected: newer.Endpoints{
				Subsets: []newer.EndpointSubset{
					{
						Ports:     []newer.EndpointPort{{Name: "", Port: 88, Protocol: newer.ProtocolUDP}},
						Addresses: []newer.EndpointAddress{{IP: "1.2.3.4"}},
					},
					{
						Ports:     []newer.EndpointPort{{Name: "", Port: 89, Protocol: newer.ProtocolUDP}},
						Addresses: []newer.EndpointAddress{{IP: "1.2.3.4"}},
					},
					{
						Ports:     []newer.EndpointPort{{Name: "", Port: 90, Protocol: newer.ProtocolUDP}},
						Addresses: []newer.EndpointAddress{{IP: "1.2.3.4"}},
					},
				}},
		},
		{
			given: current.Endpoints{
				TypeMeta: current.TypeMeta{
					ID: "one subset",
				},
				Protocol:  current.ProtocolTCP,
				Endpoints: []string{"1.2.3.4:88"},
				Subsets: []current.EndpointSubset{{
					Ports:     []current.EndpointPort{{Name: "", Port: 88, Protocol: current.ProtocolTCP}},
					Addresses: []current.EndpointAddress{{IP: "1.2.3.4"}},
				}},
			},
			expected: newer.Endpoints{
				Subsets: []newer.EndpointSubset{{
					Ports:     []newer.EndpointPort{{Name: "", Port: 88, Protocol: newer.ProtocolTCP}},
					Addresses: []newer.EndpointAddress{{IP: "1.2.3.4"}},
				}},
			},
		},
		{
			given: current.Endpoints{
				TypeMeta: current.TypeMeta{
					ID: "several subset",
				},
				Protocol:  current.ProtocolUDP,
				Endpoints: []string{"1.2.3.4:88", "5.6.7.8:88", "1.2.3.4:89", "5.6.7.8:89"},
				Subsets: []current.EndpointSubset{
					{
						Ports:     []current.EndpointPort{{Name: "", Port: 88, Protocol: current.ProtocolUDP}},
						Addresses: []current.EndpointAddress{{IP: "1.2.3.4"}, {IP: "5.6.7.8"}},
					},
					{
						Ports:     []current.EndpointPort{{Name: "", Port: 89, Protocol: current.ProtocolUDP}},
						Addresses: []current.EndpointAddress{{IP: "1.2.3.4"}, {IP: "5.6.7.8"}},
					},
					{
						Ports:     []current.EndpointPort{{Name: "named", Port: 90, Protocol: current.ProtocolUDP}},
						Addresses: []current.EndpointAddress{{IP: "1.2.3.4"}, {IP: "5.6.7.8"}},
					},
				},
			},
			expected: newer.Endpoints{
				Subsets: []newer.EndpointSubset{
					{
						Ports:     []newer.EndpointPort{{Name: "", Port: 88, Protocol: newer.ProtocolUDP}},
						Addresses: []newer.EndpointAddress{{IP: "1.2.3.4"}, {IP: "5.6.7.8"}},
					},
					{
						Ports:     []newer.EndpointPort{{Name: "", Port: 89, Protocol: newer.ProtocolUDP}},
						Addresses: []newer.EndpointAddress{{IP: "1.2.3.4"}, {IP: "5.6.7.8"}},
					},
					{
						Ports:     []newer.EndpointPort{{Name: "named", Port: 90, Protocol: newer.ProtocolUDP}},
						Addresses: []newer.EndpointAddress{{IP: "1.2.3.4"}, {IP: "5.6.7.8"}},
					},
				}},
		},
	}

	for i, tc := range testCases {
		// Convert versioned -> internal.
		got := newer.Endpoints{}
		if err := newer.Scheme.Convert(&tc.given, &got); err != nil {
			t.Errorf("[Case: %d] Unexpected error: %v", i, err)
			continue
		}
		if !newer.Semantic.DeepEqual(got.Subsets, tc.expected.Subsets) {
			t.Errorf("[Case: %d] Expected %#v, got %#v", i, tc.expected.Subsets, got.Subsets)
		}

		// Convert internal -> versioned.
		got2 := current.Endpoints{}
		if err := newer.Scheme.Convert(&got, &got2); err != nil {
			t.Errorf("[Case: %d] Unexpected error: %v", i, err)
			continue
		}
		if got2.Protocol != tc.given.Protocol || !newer.Semantic.DeepEqual(got2.Endpoints, tc.given.Endpoints) {
			t.Errorf("[Case: %d] Expected %#v, got %#v", i, tc.given.Endpoints, got2.Endpoints)
		}
	}
}

func TestSecretVolumeSourceConversion(t *testing.T) {
	given := current.SecretVolumeSource{
		Target: current.ObjectReference{
			ID: "foo",
		},
	}

	expected := newer.SecretVolumeSource{
		SecretName: "foo",
	}

	got := newer.SecretVolumeSource{}
	if err := newer.Scheme.Convert(&given, &got); err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	if got.SecretName != expected.SecretName {
		t.Errorf("Expected %v; got %v", expected, got)
	}

	got2 := current.SecretVolumeSource{}
	if err := newer.Scheme.Convert(&got, &got2); err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	if got2.Target.ID != given.Target.ID {
		t.Errorf("Expected %v; got %v", given, got2)
	}
}

func TestBadSecurityContextConversion(t *testing.T) {
	priv := false
	testCases := map[string]struct {
		c   *current.Container
		err string
	}{
		// this use case must use true for the container and false for the sc.  Otherwise the defaulter
		// will assume privileged was left undefined (since it is the default value) and copy the
		// sc setting upwards
		"mismatched privileged": {
			c: &current.Container{
				Privileged: true,
				SecurityContext: &current.SecurityContext{
					Privileged: &priv,
				},
			},
			err: "container privileged settings do not match security context settings, cannot convert",
		},
		"mismatched caps add": {
			c: &current.Container{
				Capabilities: current.Capabilities{
					Add: []current.CapabilityType{"foo"},
				},
				SecurityContext: &current.SecurityContext{
					Capabilities: &current.Capabilities{
						Add: []current.CapabilityType{"bar"},
					},
				},
			},
			err: "container capability settings do not match security context settings, cannot convert",
		},
		"mismatched caps drop": {
			c: &current.Container{
				Capabilities: current.Capabilities{
					Drop: []current.CapabilityType{"foo"},
				},
				SecurityContext: &current.SecurityContext{
					Capabilities: &current.Capabilities{
						Drop: []current.CapabilityType{"bar"},
					},
				},
			},
			err: "container capability settings do not match security context settings, cannot convert",
		},
	}

	for k, v := range testCases {
		got := newer.Container{}
		err := newer.Scheme.Convert(v.c, &got)
		if err == nil {
			t.Errorf("expected error for case %s but got none", k)
		} else {
			if err.Error() != v.err {
				t.Errorf("unexpected error for case %s.  Expected: %s but got: %s", k, v.err, err.Error())
			}
		}
	}

}
