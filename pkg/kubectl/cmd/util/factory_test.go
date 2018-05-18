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

package util

import (
	"strings"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/sets"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/kubectl"
	"k8s.io/kubernetes/pkg/kubectl/genericclioptions"
)

func TestPortsForObject(t *testing.T) {
	f := NewFactory(genericclioptions.NewTestConfigFlags())

	pod := &api.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: "baz", Namespace: "test", ResourceVersion: "12"},
		Spec: api.PodSpec{
			Containers: []api.Container{
				{
					Ports: []api.ContainerPort{
						{
							ContainerPort: 101,
						},
					},
				},
			},
		},
	}

	expected := sets.NewString("101")
	ports, err := f.PortsForObject(pod)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	got := sets.NewString(ports...)

	if !expected.Equal(got) {
		t.Fatalf("Ports mismatch! Expected %v, got %v", expected, got)
	}
}

func TestProtocolsForObject(t *testing.T) {
	f := NewFactory(genericclioptions.NewTestConfigFlags())

	pod := &api.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: "baz", Namespace: "test", ResourceVersion: "12"},
		Spec: api.PodSpec{
			Containers: []api.Container{
				{
					Ports: []api.ContainerPort{
						{
							ContainerPort: 101,
							Protocol:      api.ProtocolTCP,
						},
						{
							ContainerPort: 102,
							Protocol:      api.ProtocolUDP,
						},
					},
				},
			},
		},
	}

	expected := sets.NewString("101/TCP", "102/UDP")
	protocolsMap, err := f.ProtocolsForObject(pod)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	protocolsString := kubectl.MakeProtocols(protocolsMap)
	protocolsStrings := strings.Split(protocolsString, ",")
	got := sets.NewString(protocolsStrings...)

	if !expected.Equal(got) {
		t.Fatalf("Protocols mismatch! Expected %v, got %v", expected, got)
	}
}

func TestLabelsForObject(t *testing.T) {
	f := NewFactory(genericclioptions.NewTestConfigFlags())

	tests := []struct {
		name     string
		object   runtime.Object
		expected string
		err      error
	}{
		{
			name: "successful re-use of labels",
			object: &api.Service{
				ObjectMeta: metav1.ObjectMeta{Name: "baz", Namespace: "test", Labels: map[string]string{"svc": "test"}},
				TypeMeta:   metav1.TypeMeta{Kind: "Service", APIVersion: "v1"},
			},
			expected: "svc=test",
			err:      nil,
		},
		{
			name: "empty labels",
			object: &api.Service{
				ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: "test", Labels: map[string]string{}},
				TypeMeta:   metav1.TypeMeta{Kind: "Service", APIVersion: "v1"},
			},
			expected: "",
			err:      nil,
		},
		{
			name: "nil labels",
			object: &api.Service{
				ObjectMeta: metav1.ObjectMeta{Name: "zen", Namespace: "test", Labels: nil},
				TypeMeta:   metav1.TypeMeta{Kind: "Service", APIVersion: "v1"},
			},
			expected: "",
			err:      nil,
		},
	}

	for _, test := range tests {
		gotLabels, err := f.LabelsForObject(test.object)
		if err != test.err {
			t.Fatalf("%s: Error mismatch: Expected %v, got %v", test.name, test.err, err)
		}
		got := kubectl.MakeLabels(gotLabels)
		if test.expected != got {
			t.Fatalf("%s: Labels mismatch! Expected %s, got %s", test.name, test.expected, got)
		}

	}
}

func TestCanBeExposed(t *testing.T) {
	factory := NewFactory(genericclioptions.NewTestConfigFlags())
	tests := []struct {
		kind      schema.GroupKind
		expectErr bool
	}{
		{
			kind:      api.Kind("ReplicationController"),
			expectErr: false,
		},
		{
			kind:      api.Kind("Node"),
			expectErr: true,
		},
	}

	for _, test := range tests {
		err := factory.CanBeExposed(test.kind)
		if test.expectErr && err == nil {
			t.Error("unexpected non-error")
		}
		if !test.expectErr && err != nil {
			t.Errorf("unexpected error: %v", err)
		}
	}
}

func TestMakePortsString(t *testing.T) {
	tests := []struct {
		ports          []api.ServicePort
		useNodePort    bool
		expectedOutput string
	}{
		{ports: nil, expectedOutput: ""},
		{ports: []api.ServicePort{}, expectedOutput: ""},
		{ports: []api.ServicePort{
			{
				Port:     80,
				Protocol: "TCP",
			},
		},
			expectedOutput: "tcp:80",
		},
		{ports: []api.ServicePort{
			{
				Port:     80,
				Protocol: "TCP",
			},
			{
				Port:     8080,
				Protocol: "UDP",
			},
			{
				Port:     9000,
				Protocol: "TCP",
			},
		},
			expectedOutput: "tcp:80,udp:8080,tcp:9000",
		},
		{ports: []api.ServicePort{
			{
				Port:     80,
				NodePort: 9090,
				Protocol: "TCP",
			},
			{
				Port:     8080,
				NodePort: 80,
				Protocol: "UDP",
			},
		},
			useNodePort:    true,
			expectedOutput: "tcp:9090,udp:80",
		},
	}
	for _, test := range tests {
		output := makePortsString(test.ports, test.useNodePort)
		if output != test.expectedOutput {
			t.Errorf("expected: %s, saw: %s.", test.expectedOutput, output)
		}
	}
}
