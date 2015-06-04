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

package util

import (
	"sort"
	"testing"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client/clientcmd"
	clientcmdapi "github.com/GoogleCloudPlatform/kubernetes/pkg/client/clientcmd/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/kubectl"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
)

func TestNewFactoryDefaultFlagBindings(t *testing.T) {
	factory := NewFactory(nil)

	if !factory.flags.HasFlags() {
		t.Errorf("Expected flags, but didn't get any")
	}
}

func TestNewFactoryNoFlagBindings(t *testing.T) {
	clientConfig := clientcmd.NewDefaultClientConfig(*clientcmdapi.NewConfig(), &clientcmd.ConfigOverrides{})
	factory := NewFactory(clientConfig)

	if factory.flags.HasFlags() {
		t.Errorf("Expected zero flags, but got %v", factory.flags)
	}
}

func TestPodSelectorForObject(t *testing.T) {
	f := NewFactory(nil)

	svc := &api.Service{
		ObjectMeta: api.ObjectMeta{Name: "baz", Namespace: "test"},
		Spec: api.ServiceSpec{
			Selector: map[string]string{
				"foo": "bar",
			},
		},
	}

	expected := "foo=bar"
	got, err := f.PodSelectorForObject(svc)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	if expected != got {
		t.Fatalf("Selector mismatch! Expected %s, got %s", expected, got)
	}
}

func TestPortsForObject(t *testing.T) {
	f := NewFactory(nil)

	pod := &api.Pod{
		ObjectMeta: api.ObjectMeta{Name: "baz", Namespace: "test", ResourceVersion: "12"},
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

	expected := []string{"101"}
	got, err := f.PortsForObject(pod)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	if len(expected) != len(got) {
		t.Fatalf("Ports size mismatch! Expected %d, got %d", len(expected), len(got))
	}

	sort.Strings(expected)
	sort.Strings(got)

	for i, port := range got {
		if port != expected[i] {
			t.Fatalf("Port mismatch! Expected %s, got %s", expected[i], port)
		}
	}
}

func TestLabelsForObject(t *testing.T) {
	f := NewFactory(nil)

	tests := []struct {
		name     string
		object   runtime.Object
		expected string
		err      error
	}{
		{
			name: "successful re-use of labels",
			object: &api.Service{
				ObjectMeta: api.ObjectMeta{Name: "baz", Namespace: "test", Labels: map[string]string{"svc": "test"}},
				TypeMeta:   api.TypeMeta{Kind: "Service", APIVersion: "v1"},
			},
			expected: "svc=test",
			err:      nil,
		},
		{
			name: "empty labels",
			object: &api.Service{
				ObjectMeta: api.ObjectMeta{Name: "foo", Namespace: "test", Labels: map[string]string{}},
				TypeMeta:   api.TypeMeta{Kind: "Service", APIVersion: "v1"},
			},
			expected: "",
			err:      nil,
		},
		{
			name: "nil labels",
			object: &api.Service{
				ObjectMeta: api.ObjectMeta{Name: "zen", Namespace: "test", Labels: nil},
				TypeMeta:   api.TypeMeta{Kind: "Service", APIVersion: "v1"},
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

func TestFlagUnderscoreRenaming(t *testing.T) {
	factory := NewFactory(nil)

	factory.flags.SetNormalizeFunc(util.WordSepNormalizeFunc)
	factory.flags.Bool("valid_flag", false, "bool value")

	// In case of failure of this test check this PR: spf13/pflag#23
	if factory.flags.Lookup("valid_flag").Name != "valid-flag" {
		t.Fatalf("Expected flag name to be valid-flag, got %s", factory.flags.Lookup("valid_flag").Name)
	}
}
