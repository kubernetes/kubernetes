/*
Copyright 2017 The Kubernetes Authors.

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
	"bytes"
	"reflect"
	"sort"
	"testing"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"

	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	"k8s.io/kubernetes/cmd/kubeadm/app/constants"
)

var files = map[string][]byte{
	"foo": []byte(`
kind: Foo
apiVersion: foo.k8s.io/v1
fooField: foo
`),
	"bar": []byte(`
apiVersion: bar.k8s.io/v2
barField: bar
kind: Bar
`),
	"baz": []byte(`
apiVersion: baz.k8s.io/v1
kind: Baz
baz:
	foo: bar
`),
	"nokind": []byte(`
apiVersion: baz.k8s.io/v1
foo: foo
bar: bar
`),
	"noapiversion": []byte(`
kind: Bar
foo: foo
bar: bar
`),
}

func TestMarshalUnmarshalYaml(t *testing.T) {
	pod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "someName",
			Namespace: "testNamespace",
			Labels: map[string]string{
				"test": "yes",
			},
		},
		Spec: corev1.PodSpec{
			RestartPolicy: corev1.RestartPolicyAlways,
		},
	}

	bytes, err := MarshalToYaml(pod, corev1.SchemeGroupVersion)
	if err != nil {
		t.Fatalf("unexpected error marshalling: %v", err)
	}

	t.Logf("\n%s", bytes)

	obj2, err := UniversalUnmarshal(bytes)
	if err != nil {
		t.Fatalf("unexpected error marshalling: %v", err)
	}

	pod2, ok := obj2.(*corev1.Pod)
	if !ok {
		t.Fatal("did not get a Pod")
	}

	if pod2.Name != pod.Name {
		t.Errorf("expected %q, got %q", pod.Name, pod2.Name)
	}

	if pod2.Namespace != pod.Namespace {
		t.Errorf("expected %q, got %q", pod.Namespace, pod2.Namespace)
	}

	if !reflect.DeepEqual(pod2.Labels, pod.Labels) {
		t.Errorf("expected %v, got %v", pod.Labels, pod2.Labels)
	}

	if pod2.Spec.RestartPolicy != pod.Spec.RestartPolicy {
		t.Errorf("expected %q, got %q", pod.Spec.RestartPolicy, pod2.Spec.RestartPolicy)
	}
}

func TestUnmarshalJson(t *testing.T) {
	bytes := []byte(string(`{
  "apiVersion": "v1",
  "kind": "Pod",
  "metadata": {
    "name": "someName",
    "namespace": "testNamespace",
	"labels": {
		"test": "yes"
	}
  },
  "spec": {
	"restartPolicy": "Always"
  }
}`))

	t.Logf("\n%s", bytes)

	obj2, err := UniversalUnmarshal(bytes)
	if err != nil {
		t.Fatalf("unexpected error marshalling: %v", err)
	}

	pod2, ok := obj2.(*corev1.Pod)
	if !ok {
		t.Fatal("did not get a Pod")
	}

	if pod2.Name != "someName" {
		t.Errorf("expected someName, got %q", pod2.Name)
	}

	if pod2.Namespace != "testNamespace" {
		t.Errorf("expected testNamespace, got %q", pod2.Namespace)
	}

	if !reflect.DeepEqual(pod2.Labels, map[string]string{"test": "yes"}) {
		t.Errorf("expected [test:yes], got %v", pod2.Labels)
	}

	if pod2.Spec.RestartPolicy != "Always" {
		t.Errorf("expected Always, got %q", pod2.Spec.RestartPolicy)
	}
}

func TestSplitYAMLDocuments(t *testing.T) {
	var tests = []struct {
		name         string
		fileContents []byte
		gvkmap       kubeadmapi.DocumentMap
		expectedErr  bool
	}{
		{
			name:         "FooOnly",
			fileContents: files["foo"],
			gvkmap: kubeadmapi.DocumentMap{
				{Group: "foo.k8s.io", Version: "v1", Kind: "Foo"}: files["foo"],
			},
		},
		{
			name:         "FooBar",
			fileContents: bytes.Join([][]byte{files["foo"], files["bar"]}, []byte(constants.YAMLDocumentSeparator)),
			gvkmap: kubeadmapi.DocumentMap{
				{Group: "foo.k8s.io", Version: "v1", Kind: "Foo"}: files["foo"],
				{Group: "bar.k8s.io", Version: "v2", Kind: "Bar"}: files["bar"],
			},
		},
		{
			name:         "FooTwiceInvalid",
			fileContents: bytes.Join([][]byte{files["foo"], files["bar"], files["foo"]}, []byte(constants.YAMLDocumentSeparator)),
			expectedErr:  true,
		},
		{
			name:         "InvalidBaz",
			fileContents: bytes.Join([][]byte{files["foo"], files["baz"]}, []byte(constants.YAMLDocumentSeparator)),
			expectedErr:  true,
		},
		{
			name:         "InvalidNoKind",
			fileContents: files["nokind"],
			expectedErr:  true,
		},
		{
			name:         "InvalidNoAPIVersion",
			fileContents: files["noapiversion"],
			expectedErr:  true,
		},
	}

	for _, rt := range tests {
		t.Run(rt.name, func(t2 *testing.T) {

			gvkmap, err := SplitConfigDocuments(rt.fileContents)
			if (err != nil) != rt.expectedErr {
				t2.Errorf("expected error: %t, actual: %t", rt.expectedErr, err != nil)
			}

			if !reflect.DeepEqual(gvkmap, rt.gvkmap) {
				t2.Errorf("expected gvkmap: %s\n\tactual: %s\n", rt.gvkmap, gvkmap)
			}
		})
	}
}

func TestGroupVersionKindsFromBytes(t *testing.T) {
	var tests = []struct {
		name         string
		fileContents []byte
		gvks         []string
		expectedErr  bool
	}{
		{
			name:         "FooOnly",
			fileContents: files["foo"],
			gvks: []string{
				"foo.k8s.io/v1, Kind=Foo",
			},
		},
		{
			name:         "FooBar",
			fileContents: bytes.Join([][]byte{files["foo"], files["bar"]}, []byte(constants.YAMLDocumentSeparator)),
			gvks: []string{
				"foo.k8s.io/v1, Kind=Foo",
				"bar.k8s.io/v2, Kind=Bar",
			},
		},
		{
			name:         "FooTwiceInvalid",
			fileContents: bytes.Join([][]byte{files["foo"], files["bar"], files["foo"]}, []byte(constants.YAMLDocumentSeparator)),
			gvks:         []string{},
			expectedErr:  true,
		},
		{
			name:         "InvalidBaz",
			fileContents: bytes.Join([][]byte{files["foo"], files["baz"]}, []byte(constants.YAMLDocumentSeparator)),
			gvks:         []string{},
			expectedErr:  true,
		},
		{
			name:         "InvalidNoKind",
			fileContents: files["nokind"],
			gvks:         []string{},
			expectedErr:  true,
		},
		{
			name:         "InvalidNoAPIVersion",
			fileContents: files["noapiversion"],
			gvks:         []string{},
			expectedErr:  true,
		},
	}

	for _, rt := range tests {
		t.Run(rt.name, func(t2 *testing.T) {

			gvks, err := GroupVersionKindsFromBytes(rt.fileContents)
			if (err != nil) != rt.expectedErr {
				t2.Errorf("expected error: %t, actual: %t", rt.expectedErr, err != nil)
			}

			strgvks := []string{}
			for _, gvk := range gvks {
				strgvks = append(strgvks, gvk.String())
			}
			sort.Strings(strgvks)
			sort.Strings(rt.gvks)

			if !reflect.DeepEqual(strgvks, rt.gvks) {
				t2.Errorf("expected gvks: %s\n\tactual: %s\n", rt.gvks, strgvks)
			}
		})
	}
}

func TestGroupVersionKindsHasKind(t *testing.T) {
	var tests = []struct {
		name     string
		gvks     []schema.GroupVersionKind
		kind     string
		expected bool
	}{
		{
			name: "FooOnly",
			gvks: []schema.GroupVersionKind{
				{Group: "foo.k8s.io", Version: "v1", Kind: "Foo"},
			},
			kind:     "Foo",
			expected: true,
		},
		{
			name: "FooBar",
			gvks: []schema.GroupVersionKind{
				{Group: "foo.k8s.io", Version: "v1", Kind: "Foo"},
				{Group: "bar.k8s.io", Version: "v2", Kind: "Bar"},
			},
			kind:     "Bar",
			expected: true,
		},
		{
			name: "FooBazNoBaz",
			gvks: []schema.GroupVersionKind{
				{Group: "foo.k8s.io", Version: "v1", Kind: "Foo"},
				{Group: "bar.k8s.io", Version: "v2", Kind: "Bar"},
			},
			kind:     "Baz",
			expected: false,
		},
	}

	for _, rt := range tests {
		t.Run(rt.name, func(t2 *testing.T) {

			actual := GroupVersionKindsHasKind(rt.gvks, rt.kind)
			if rt.expected != actual {
				t2.Errorf("expected gvks has kind: %t\n\tactual: %t\n", rt.expected, actual)
			}
		})
	}
}

func TestGroupVersionKindsHasInitConfiguration(t *testing.T) {
	var tests = []struct {
		name     string
		gvks     []schema.GroupVersionKind
		kind     string
		expected bool
	}{
		{
			name: "NoInitConfiguration",
			gvks: []schema.GroupVersionKind{
				{Group: "foo.k8s.io", Version: "v1", Kind: "Foo"},
			},
			expected: false,
		},
		{
			name: "InitConfigurationFound",
			gvks: []schema.GroupVersionKind{
				{Group: "foo.k8s.io", Version: "v1", Kind: "Foo"},
				{Group: "bar.k8s.io", Version: "v2", Kind: "InitConfiguration"},
			},
			expected: true,
		},
	}

	for _, rt := range tests {
		t.Run(rt.name, func(t2 *testing.T) {

			actual := GroupVersionKindsHasInitConfiguration(rt.gvks...)
			if rt.expected != actual {
				t2.Errorf("expected gvks has InitConfiguration: %t\n\tactual: %t\n", rt.expected, actual)
			}
		})
	}
}

func TestGroupVersionKindsHasJoinConfiguration(t *testing.T) {
	var tests = []struct {
		name     string
		gvks     []schema.GroupVersionKind
		kind     string
		expected bool
	}{
		{
			name: "NoJoinConfiguration",
			gvks: []schema.GroupVersionKind{
				{Group: "foo.k8s.io", Version: "v1", Kind: "Foo"},
			},
			expected: false,
		},
		{
			name: "JoinConfigurationFound",
			gvks: []schema.GroupVersionKind{
				{Group: "foo.k8s.io", Version: "v1", Kind: "Foo"},
				{Group: "bar.k8s.io", Version: "v2", Kind: "JoinConfiguration"},
			},
			expected: true,
		},
	}

	for _, rt := range tests {
		t.Run(rt.name, func(t2 *testing.T) {

			actual := GroupVersionKindsHasJoinConfiguration(rt.gvks...)
			if rt.expected != actual {
				t2.Errorf("expected gvks has JoinConfiguration: %t\n\tactual: %t\n", rt.expected, actual)
			}
		})
	}
}

func TestGroupVersionKindsHasResetConfiguration(t *testing.T) {
	var tests = []struct {
		name     string
		gvks     []schema.GroupVersionKind
		kind     string
		expected bool
	}{
		{
			name: "NoResetConfiguration",
			gvks: []schema.GroupVersionKind{
				{Group: "foo.k8s.io", Version: "v1", Kind: "Foo"},
			},
			expected: false,
		},
		{
			name: "ResetConfigurationFound",
			gvks: []schema.GroupVersionKind{
				{Group: "foo.k8s.io", Version: "v1", Kind: "Foo"},
				{Group: "bar.k8s.io", Version: "v2", Kind: "ResetConfiguration"},
			},
			expected: true,
		},
	}

	for _, rt := range tests {
		t.Run(rt.name, func(t2 *testing.T) {

			actual := GroupVersionKindsHasResetConfiguration(rt.gvks...)
			if rt.expected != actual {
				t2.Errorf("expected gvks has ResetConfiguration: %t\n\tactual: %t\n", rt.expected, actual)
			}
		})
	}
}

func TestGroupVersionKindsHasClusterConfiguration(t *testing.T) {
	tests := []struct {
		name     string
		gvks     []schema.GroupVersionKind
		expected bool
	}{
		{
			name: "does not have ClusterConfiguration",
			gvks: []schema.GroupVersionKind{
				{Group: "foo.k8s.io", Version: "v1", Kind: "Foo"},
			},
			expected: false,
		},
		{
			name: "has ClusterConfiguration",
			gvks: []schema.GroupVersionKind{
				{Group: "foo.k8s.io", Version: "v1", Kind: "Foo"},
				{Group: "foo.k8s.io", Version: "v1", Kind: "ClusterConfiguration"},
			},
			expected: true,
		},
	}
	for _, rt := range tests {
		t.Run(rt.name, func(t *testing.T) {
			actual := GroupVersionKindsHasClusterConfiguration(rt.gvks...)
			if rt.expected != actual {
				t.Errorf("expected gvks to have a ClusterConfiguration: %t\n\tactual: %t\n", rt.expected, actual)
			}
		})
	}
}

func TestGroupVersionKindsHasUpgradeConfiguration(t *testing.T) {
	var tests = []struct {
		name     string
		gvks     []schema.GroupVersionKind
		kind     string
		expected bool
	}{
		{
			name: "no UpgradeConfiguration found",
			gvks: []schema.GroupVersionKind{
				{Group: "foo.k8s.io", Version: "v1", Kind: "Foo"},
			},
			expected: false,
		},
		{
			name: "UpgradeConfiguration is found",
			gvks: []schema.GroupVersionKind{
				{Group: "foo.k8s.io", Version: "v1", Kind: "Foo"},
				{Group: "bar.k8s.io", Version: "v2", Kind: "UpgradeConfiguration"},
			},
			expected: true,
		},
	}

	for _, rt := range tests {
		t.Run(rt.name, func(t2 *testing.T) {
			actual := GroupVersionKindsHasUpgradeConfiguration(rt.gvks...)
			if rt.expected != actual {
				t2.Errorf("expected gvks has UpgradeConfiguration: %t\n\tactual: %t\n", rt.expected, actual)
			}
		})
	}
}
