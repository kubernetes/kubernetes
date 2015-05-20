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

package v1beta1_test

import (
	"encoding/json"
	"reflect"
	"testing"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/resource"
	versioned "github.com/GoogleCloudPlatform/kubernetes/pkg/api/v1beta1"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
)

var Convert = api.Scheme.Convert

func TestEmptyObjectConversion(t *testing.T) {
	s, err := versioned.Codec.Encode(&versioned.LimitRange{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	// DeletionTimestamp is not included, while CreationTimestamp is (would always be set)
	if string(s) != `{"kind":"LimitRange","creationTimestamp":null,"apiVersion":"v1beta1","spec":{"limits":null}}` {
		t.Errorf("unexpected empty object: %s", string(s))
	}
}

func TestNodeConversion(t *testing.T) {
	version, kind, err := api.Scheme.ObjectVersionAndKind(&versioned.Minion{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if version != "v1beta1" || kind != "Minion" {
		t.Errorf("unexpected version and kind: %s %s", version, kind)
	}

	api.Scheme.Log(t)
	obj, err := versioned.Codec.Decode([]byte(`{"kind":"Node","apiVersion":"v1beta1"}`))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if _, ok := obj.(*api.Node); !ok {
		t.Errorf("unexpected type: %#v", obj)
	}

	obj, err = versioned.Codec.Decode([]byte(`{"kind":"NodeList","apiVersion":"v1beta1"}`))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if _, ok := obj.(*api.NodeList); !ok {
		t.Errorf("unexpected type: %#v", obj)
	}

	obj = &api.Node{}
	if err := versioned.Codec.DecodeInto([]byte(`{"kind":"Node","apiVersion":"v1beta1"}`), obj); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	obj = &api.Node{}
	data, err := versioned.Codec.Encode(obj)
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

func TestEnvConversion(t *testing.T) {
	nonCanonical := []versioned.EnvVar{
		{Key: "EV"},
		{Key: "EV", Name: "EX"},
	}
	canonical := []api.EnvVar{
		{Name: "EV"},
		{Name: "EX"},
	}
	for i := range nonCanonical {
		var got api.EnvVar
		err := Convert(&nonCanonical[i], &got)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if e, a := canonical[i], got; !reflect.DeepEqual(e, a) {
			t.Errorf("expected %v, got %v", e, a)
		}
	}

	// Test conversion the other way, too.
	for i := range canonical {
		var got versioned.EnvVar
		err := Convert(&canonical[i], &got)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if e, a := canonical[i].Name, got.Key; e != a {
			t.Errorf("expected %v, got %v", e, a)
		}
		if e, a := canonical[i].Name, got.Name; e != a {
			t.Errorf("expected %v, got %v", e, a)
		}
	}
}

func TestVolumeMountConversionToOld(t *testing.T) {
	table := []struct {
		in  api.VolumeMount
		out versioned.VolumeMount
	}{
		{
			in:  api.VolumeMount{Name: "foo", MountPath: "/dev/foo", ReadOnly: true},
			out: versioned.VolumeMount{Name: "foo", MountPath: "/dev/foo", Path: "/dev/foo", ReadOnly: true},
		},
	}
	for _, item := range table {
		got := versioned.VolumeMount{}
		err := Convert(&item.in, &got)
		if err != nil {
			t.Errorf("Unexpected error: %v", err)
			continue
		}
		if e, a := item.out, got; !reflect.DeepEqual(e, a) {
			t.Errorf("Expected: %#v, got %#v", e, a)
		}
	}
}

func TestVolumeMountConversionToNew(t *testing.T) {
	table := []struct {
		in  versioned.VolumeMount
		out api.VolumeMount
	}{
		{
			in:  versioned.VolumeMount{Name: "foo", MountPath: "/dev/foo", ReadOnly: true},
			out: api.VolumeMount{Name: "foo", MountPath: "/dev/foo", ReadOnly: true},
		}, {
			in:  versioned.VolumeMount{Name: "foo", MountPath: "/dev/foo", Path: "/dev/bar", ReadOnly: true},
			out: api.VolumeMount{Name: "foo", MountPath: "/dev/foo", ReadOnly: true},
		}, {
			in:  versioned.VolumeMount{Name: "foo", Path: "/dev/bar", ReadOnly: true},
			out: api.VolumeMount{Name: "foo", MountPath: "/dev/bar", ReadOnly: true},
		},
	}
	for _, item := range table {
		got := api.VolumeMount{}
		err := Convert(&item.in, &got)
		if err != nil {
			t.Errorf("Unexpected error: %v", err)
			continue
		}
		if e, a := item.out, got; !reflect.DeepEqual(e, a) {
			t.Errorf("Expected: %#v, got %#v", e, a)
		}
	}
}

func TestMinionListConversionToNew(t *testing.T) {
	oldMinion := func(id string) versioned.Minion {
		return versioned.Minion{
			TypeMeta:   versioned.TypeMeta{ID: id},
			ExternalID: id}
	}
	newNode := func(id string) api.Node {
		return api.Node{
			ObjectMeta: api.ObjectMeta{Name: id},
			Spec:       api.NodeSpec{ExternalID: id},
		}
	}
	oldMinions := []versioned.Minion{
		oldMinion("foo"),
		oldMinion("bar"),
	}
	newMinions := []api.Node{
		newNode("foo"),
		newNode("bar"),
	}

	table := []struct {
		oldML *versioned.MinionList
		newML *api.NodeList
	}{
		{
			oldML: &versioned.MinionList{Items: oldMinions},
			newML: &api.NodeList{Items: newMinions},
		}, {
			oldML: &versioned.MinionList{Minions: oldMinions},
			newML: &api.NodeList{Items: newMinions},
		}, {
			oldML: &versioned.MinionList{
				Items:   oldMinions,
				Minions: []versioned.Minion{oldMinion("baz")},
			},
			newML: &api.NodeList{Items: newMinions},
		},
	}

	for _, item := range table {
		got := &api.NodeList{}
		err := Convert(item.oldML, got)
		if err != nil {
			t.Errorf("Unexpected error: %v", err)
		}
		if e, a := item.newML, got; !api.Semantic.DeepEqual(e, a) {
			t.Errorf("Expected: %#v, got %#v", e, a)
		}
	}
}

func TestMinionListConversionToOld(t *testing.T) {
	oldMinion := func(id string) versioned.Minion {
		return versioned.Minion{TypeMeta: versioned.TypeMeta{ID: id}}
	}
	newNode := func(id string) api.Node {
		return api.Node{ObjectMeta: api.ObjectMeta{Name: id}}
	}
	oldMinions := []versioned.Minion{
		oldMinion("foo"),
		oldMinion("bar"),
	}
	newMinions := []api.Node{
		newNode("foo"),
		newNode("bar"),
	}

	newML := &api.NodeList{Items: newMinions}
	oldML := &versioned.MinionList{
		Items:   oldMinions,
		Minions: oldMinions,
	}

	got := &versioned.MinionList{}
	err := Convert(newML, got)
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	if e, a := oldML, got; !api.Semantic.DeepEqual(e, a) {
		t.Errorf("Expected: %#v, got %#v", e, a)
	}
}

func TestServiceEmptySelector(t *testing.T) {
	// Nil map should be preserved
	svc := &versioned.Service{Selector: nil}
	data, err := api.Scheme.EncodeToVersion(svc, "v1beta1")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	obj, err := api.Scheme.Decode(data)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	selector := obj.(*api.Service).Spec.Selector
	if selector != nil {
		t.Errorf("unexpected selector: %#v", obj)
	}

	// Empty map should be preserved
	svc2 := &versioned.Service{Selector: map[string]string{}}
	data, err = api.Scheme.EncodeToVersion(svc2, "v1beta1")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	obj, err = api.Scheme.Decode(data)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	selector = obj.(*api.Service).Spec.Selector
	if selector == nil || len(selector) != 0 {
		t.Errorf("unexpected selector: %#v", obj)
	}
}

func TestServicePorts(t *testing.T) {
	testCases := []struct {
		given     versioned.Service
		expected  api.Service
		roundtrip versioned.Service
	}{
		{
			given: versioned.Service{
				TypeMeta: versioned.TypeMeta{
					ID: "legacy-with-defaults",
				},
				Port:     111,
				Protocol: versioned.ProtocolTCP,
			},
			expected: api.Service{
				Spec: api.ServiceSpec{Ports: []api.ServicePort{{
					Port:     111,
					Protocol: api.ProtocolTCP,
				}}},
			},
			roundtrip: versioned.Service{
				Ports: []versioned.ServicePort{{
					Port:     111,
					Protocol: versioned.ProtocolTCP,
				}},
			},
		},
		{
			given: versioned.Service{
				TypeMeta: versioned.TypeMeta{
					ID: "legacy-full",
				},
				PortName:      "p",
				Port:          111,
				Protocol:      versioned.ProtocolTCP,
				ContainerPort: util.NewIntOrStringFromString("p"),
			},
			expected: api.Service{
				Spec: api.ServiceSpec{Ports: []api.ServicePort{{
					Name:       "p",
					Port:       111,
					Protocol:   api.ProtocolTCP,
					TargetPort: util.NewIntOrStringFromString("p"),
				}}},
			},
			roundtrip: versioned.Service{
				Ports: []versioned.ServicePort{{
					Name:          "p",
					Port:          111,
					Protocol:      versioned.ProtocolTCP,
					ContainerPort: util.NewIntOrStringFromString("p"),
				}},
			},
		},
		{
			given: versioned.Service{
				TypeMeta: versioned.TypeMeta{
					ID: "both",
				},
				PortName:      "p",
				Port:          111,
				Protocol:      versioned.ProtocolTCP,
				ContainerPort: util.NewIntOrStringFromString("p"),
				Ports: []versioned.ServicePort{{
					Name:          "q",
					Port:          222,
					Protocol:      versioned.ProtocolUDP,
					ContainerPort: util.NewIntOrStringFromInt(93),
				}},
			},
			expected: api.Service{
				Spec: api.ServiceSpec{Ports: []api.ServicePort{{
					Name:       "q",
					Port:       222,
					Protocol:   api.ProtocolUDP,
					TargetPort: util.NewIntOrStringFromInt(93),
				}}},
			},
			roundtrip: versioned.Service{
				Ports: []versioned.ServicePort{{
					Name:          "q",
					Port:          222,
					Protocol:      versioned.ProtocolUDP,
					ContainerPort: util.NewIntOrStringFromInt(93),
				}},
			},
		},
		{
			given: versioned.Service{
				TypeMeta: versioned.TypeMeta{
					ID: "one",
				},
				Ports: []versioned.ServicePort{{
					Name:          "p",
					Port:          111,
					Protocol:      versioned.ProtocolUDP,
					ContainerPort: util.NewIntOrStringFromInt(93),
				}},
			},
			expected: api.Service{
				Spec: api.ServiceSpec{Ports: []api.ServicePort{{
					Name:       "p",
					Port:       111,
					Protocol:   api.ProtocolUDP,
					TargetPort: util.NewIntOrStringFromInt(93),
				}}},
			},
			roundtrip: versioned.Service{
				Ports: []versioned.ServicePort{{
					Name:          "p",
					Port:          111,
					Protocol:      versioned.ProtocolUDP,
					ContainerPort: util.NewIntOrStringFromInt(93),
				}},
			},
		},
		{
			given: versioned.Service{
				TypeMeta: versioned.TypeMeta{
					ID: "two",
				},
				Ports: []versioned.ServicePort{{
					Name:          "p",
					Port:          111,
					Protocol:      versioned.ProtocolUDP,
					ContainerPort: util.NewIntOrStringFromInt(93),
				}, {
					Name:          "q",
					Port:          222,
					Protocol:      versioned.ProtocolTCP,
					ContainerPort: util.NewIntOrStringFromInt(76),
				}},
			},
			expected: api.Service{
				Spec: api.ServiceSpec{Ports: []api.ServicePort{{
					Name:       "p",
					Port:       111,
					Protocol:   api.ProtocolUDP,
					TargetPort: util.NewIntOrStringFromInt(93),
				}, {
					Name:       "q",
					Port:       222,
					Protocol:   api.ProtocolTCP,
					TargetPort: util.NewIntOrStringFromInt(76),
				}}},
			},
			roundtrip: versioned.Service{
				Ports: []versioned.ServicePort{{
					Name:          "p",
					Port:          111,
					Protocol:      versioned.ProtocolUDP,
					ContainerPort: util.NewIntOrStringFromInt(93),
				}, {
					Name:          "q",
					Port:          222,
					Protocol:      versioned.ProtocolTCP,
					ContainerPort: util.NewIntOrStringFromInt(76),
				}},
			},
		},
	}

	for i, tc := range testCases {
		// Convert versioned -> internal.
		got := api.Service{}
		if err := Convert(&tc.given, &got); err != nil {
			t.Errorf("[Case: %d] Unexpected error: %v", i, err)
			continue
		}
		if !reflect.DeepEqual(got.Spec.Ports, tc.expected.Spec.Ports) {
			t.Errorf("[Case: %d] Expected %v, got %v", i, tc.expected.Spec.Ports, got.Spec.Ports)
		}

		// Convert internal -> versioned.
		got2 := versioned.Service{}
		if err := Convert(&got, &got2); err != nil {
			t.Errorf("[Case: %d] Unexpected error: %v", i, err)
			continue
		}
		if !reflect.DeepEqual(got2.Ports, tc.roundtrip.Ports) {
			t.Errorf("[Case: %d] Expected %v, got %v", i, tc.roundtrip.Ports, got2.Ports)
		}
	}
}

func TestPullPolicyConversion(t *testing.T) {
	table := []struct {
		versioned versioned.PullPolicy
		internal  api.PullPolicy
	}{
		{
			versioned: versioned.PullAlways,
			internal:  api.PullAlways,
		}, {
			versioned: versioned.PullNever,
			internal:  api.PullNever,
		}, {
			versioned: versioned.PullIfNotPresent,
			internal:  api.PullIfNotPresent,
		}, {
			versioned: "",
			internal:  "",
		}, {
			versioned: "invalid value",
			internal:  "invalid value",
		},
	}
	for _, item := range table {
		var got api.PullPolicy
		err := Convert(&item.versioned, &got)
		if err != nil {
			t.Errorf("Unexpected error: %v", err)
			continue
		}
		if e, a := item.internal, got; e != a {
			t.Errorf("Expected: %q, got %q", e, a)
		}
	}
	for _, item := range table {
		var got versioned.PullPolicy
		err := Convert(&item.internal, &got)
		if err != nil {
			t.Errorf("Unexpected error: %v", err)
			continue
		}
		if e, a := item.versioned, got; e != a {
			t.Errorf("Expected: %q, got %q", e, a)
		}
	}
}

func getResourceRequirements(cpu, memory resource.Quantity) versioned.ResourceRequirements {
	res := versioned.ResourceRequirements{}
	res.Limits = versioned.ResourceList{}
	if cpu.Value() > 0 {
		res.Limits[versioned.ResourceCPU] = util.NewIntOrStringFromInt(int(cpu.Value()))
	}
	if memory.Value() > 0 {
		res.Limits[versioned.ResourceMemory] = util.NewIntOrStringFromInt(int(memory.Value()))
	}

	return res
}

func TestContainerConversion(t *testing.T) {
	cpuLimit := resource.MustParse("10")
	memoryLimit := resource.MustParse("10M")
	null := resource.Quantity{}
	testCases := []versioned.Container{
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
		got := api.Container{}
		if err := Convert(&tc, &got); err != nil {
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
		given    versioned.Endpoints
		expected api.Endpoints
	}{
		{
			given: versioned.Endpoints{
				TypeMeta: versioned.TypeMeta{
					ID: "empty",
				},
				Protocol:  versioned.ProtocolTCP,
				Endpoints: []string{},
			},
			expected: api.Endpoints{
				Subsets: []api.EndpointSubset{},
			},
		},
		{
			given: versioned.Endpoints{
				TypeMeta: versioned.TypeMeta{
					ID: "one legacy",
				},
				Protocol:  versioned.ProtocolTCP,
				Endpoints: []string{"1.2.3.4:88"},
			},
			expected: api.Endpoints{
				Subsets: []api.EndpointSubset{{
					Ports:     []api.EndpointPort{{Name: "", Port: 88, Protocol: api.ProtocolTCP}},
					Addresses: []api.EndpointAddress{{IP: "1.2.3.4"}},
				}},
			},
		},
		{
			given: versioned.Endpoints{
				TypeMeta: versioned.TypeMeta{
					ID: "several legacy",
				},
				Protocol:  versioned.ProtocolUDP,
				Endpoints: []string{"1.2.3.4:88", "1.2.3.4:89", "1.2.3.4:90"},
			},
			expected: api.Endpoints{
				Subsets: []api.EndpointSubset{
					{
						Ports:     []api.EndpointPort{{Name: "", Port: 88, Protocol: api.ProtocolUDP}},
						Addresses: []api.EndpointAddress{{IP: "1.2.3.4"}},
					},
					{
						Ports:     []api.EndpointPort{{Name: "", Port: 89, Protocol: api.ProtocolUDP}},
						Addresses: []api.EndpointAddress{{IP: "1.2.3.4"}},
					},
					{
						Ports:     []api.EndpointPort{{Name: "", Port: 90, Protocol: api.ProtocolUDP}},
						Addresses: []api.EndpointAddress{{IP: "1.2.3.4"}},
					},
				}},
		},
		{
			given: versioned.Endpoints{
				TypeMeta: versioned.TypeMeta{
					ID: "one subset",
				},
				Protocol:  versioned.ProtocolTCP,
				Endpoints: []string{"1.2.3.4:88"},
				Subsets: []versioned.EndpointSubset{{
					Ports:     []versioned.EndpointPort{{Name: "", Port: 88, Protocol: versioned.ProtocolTCP}},
					Addresses: []versioned.EndpointAddress{{IP: "1.2.3.4"}},
				}},
			},
			expected: api.Endpoints{
				Subsets: []api.EndpointSubset{{
					Ports:     []api.EndpointPort{{Name: "", Port: 88, Protocol: api.ProtocolTCP}},
					Addresses: []api.EndpointAddress{{IP: "1.2.3.4"}},
				}},
			},
		},
		{
			given: versioned.Endpoints{
				TypeMeta: versioned.TypeMeta{
					ID: "several subset",
				},
				Protocol:  versioned.ProtocolUDP,
				Endpoints: []string{"1.2.3.4:88", "5.6.7.8:88", "1.2.3.4:89", "5.6.7.8:89"},
				Subsets: []versioned.EndpointSubset{
					{
						Ports:     []versioned.EndpointPort{{Name: "", Port: 88, Protocol: versioned.ProtocolUDP}},
						Addresses: []versioned.EndpointAddress{{IP: "1.2.3.4"}, {IP: "5.6.7.8"}},
					},
					{
						Ports:     []versioned.EndpointPort{{Name: "", Port: 89, Protocol: versioned.ProtocolUDP}},
						Addresses: []versioned.EndpointAddress{{IP: "1.2.3.4"}, {IP: "5.6.7.8"}},
					},
					{
						Ports:     []versioned.EndpointPort{{Name: "named", Port: 90, Protocol: versioned.ProtocolUDP}},
						Addresses: []versioned.EndpointAddress{{IP: "1.2.3.4"}, {IP: "5.6.7.8"}},
					},
				},
			},
			expected: api.Endpoints{
				Subsets: []api.EndpointSubset{
					{
						Ports:     []api.EndpointPort{{Name: "", Port: 88, Protocol: api.ProtocolUDP}},
						Addresses: []api.EndpointAddress{{IP: "1.2.3.4"}, {IP: "5.6.7.8"}},
					},
					{
						Ports:     []api.EndpointPort{{Name: "", Port: 89, Protocol: api.ProtocolUDP}},
						Addresses: []api.EndpointAddress{{IP: "1.2.3.4"}, {IP: "5.6.7.8"}},
					},
					{
						Ports:     []api.EndpointPort{{Name: "named", Port: 90, Protocol: api.ProtocolUDP}},
						Addresses: []api.EndpointAddress{{IP: "1.2.3.4"}, {IP: "5.6.7.8"}},
					},
				}},
		},
	}

	for i, tc := range testCases {
		// Convert versioned -> internal.
		got := api.Endpoints{}
		if err := Convert(&tc.given, &got); err != nil {
			t.Errorf("[Case: %d] Unexpected error: %v", i, err)
			continue
		}
		if !api.Semantic.DeepEqual(got.Subsets, tc.expected.Subsets) {
			t.Errorf("[Case: %d] Expected %#v, got %#v", i, tc.expected.Subsets, got.Subsets)
		}

		// Convert internal -> versioned.
		got2 := versioned.Endpoints{}
		if err := Convert(&got, &got2); err != nil {
			t.Errorf("[Case: %d] Unexpected error: %v", i, err)
			continue
		}
		if got2.Protocol != tc.given.Protocol || !api.Semantic.DeepEqual(got2.Endpoints, tc.given.Endpoints) {
			t.Errorf("[Case: %d] Expected %s %#v, got %s %#v", i, tc.given.Protocol, tc.given.Endpoints, got2.Protocol, got2.Endpoints)
		}
	}
}

func TestSecretVolumeSourceConversion(t *testing.T) {
	given := versioned.SecretVolumeSource{
		Target: versioned.ObjectReference{
			ID: "foo",
		},
	}

	expected := api.SecretVolumeSource{
		SecretName: "foo",
	}

	got := api.SecretVolumeSource{}
	if err := Convert(&given, &got); err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	if got.SecretName != expected.SecretName {
		t.Errorf("Expected %v; got %v", expected, got)
	}

	got2 := versioned.SecretVolumeSource{}
	if err := Convert(&got, &got2); err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	if got2.Target.ID != given.Target.ID {
		t.Errorf("Expected %v; got %v", given, got2)
	}
}

func TestBadSecurityContextConversion(t *testing.T) {
	priv := false
	testCases := map[string]struct {
		c   *versioned.Container
		err string
	}{
		// this use case must use true for the container and false for the sc.  Otherwise the defaulter
		// will assume privileged was left undefined (since it is the default value) and copy the
		// sc setting upwards
		"mismatched privileged": {
			c: &versioned.Container{
				Privileged: true,
				SecurityContext: &versioned.SecurityContext{
					Privileged: &priv,
				},
			},
			err: "container privileged settings do not match security context settings, cannot convert",
		},
		"mismatched caps add": {
			c: &versioned.Container{
				Capabilities: versioned.Capabilities{
					Add: []versioned.Capability{"foo"},
				},
				SecurityContext: &versioned.SecurityContext{
					Capabilities: &versioned.Capabilities{
						Add: []versioned.Capability{"bar"},
					},
				},
			},
			err: "container capability settings do not match security context settings, cannot convert",
		},
		"mismatched caps drop": {
			c: &versioned.Container{
				Capabilities: versioned.Capabilities{
					Drop: []versioned.Capability{"foo"},
				},
				SecurityContext: &versioned.SecurityContext{
					Capabilities: &versioned.Capabilities{
						Drop: []versioned.Capability{"bar"},
					},
				},
			},
			err: "container capability settings do not match security context settings, cannot convert",
		},
	}

	for k, v := range testCases {
		got := api.Container{}
		err := Convert(v.c, &got)
		if err == nil {
			t.Errorf("expected error for case %s but got none", k)
		} else {
			if err.Error() != v.err {
				t.Errorf("unexpected error for case %s.  Expected: %s but got: %s", k, v.err, err.Error())
			}
		}
	}

}
