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
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	kubeadmapiv1beta1 "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/v1beta1"
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

	obj2, err := UnmarshalFromYaml(bytes, corev1.SchemeGroupVersion)
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

func TestMarshalUnmarshalToYamlForCodecs(t *testing.T) {
	cfg := &kubeadmapiv1beta1.InitConfiguration{
		TypeMeta: metav1.TypeMeta{
			Kind:       constants.InitConfigurationKind,
			APIVersion: kubeadmapiv1beta1.SchemeGroupVersion.String(),
		},
		NodeRegistration: kubeadmapiv1beta1.NodeRegistrationOptions{
			Name:      "testNode",
			CRISocket: "/var/run/cri.sock",
		},
		BootstrapTokens: []kubeadmapiv1beta1.BootstrapToken{
			{
				Token: &kubeadmapiv1beta1.BootstrapTokenString{ID: "abcdef", Secret: "abcdef0123456789"},
			},
		},
		// NOTE: Using MarshalToYamlForCodecs and UnmarshalFromYamlForCodecs for ClusterConfiguration fields here won't work
		// by design. This is because we have a `json:"-"` annotation in order to avoid struct duplication. See the comment
		// at the kubeadmapiv1beta1.InitConfiguration definition.
	}

	kubeadmapiv1beta1.SetDefaults_InitConfiguration(cfg)
	scheme := runtime.NewScheme()
	if err := kubeadmapiv1beta1.AddToScheme(scheme); err != nil {
		t.Fatal(err)
	}
	codecs := serializer.NewCodecFactory(scheme)

	bytes, err := MarshalToYamlForCodecs(cfg, kubeadmapiv1beta1.SchemeGroupVersion, codecs)
	if err != nil {
		t.Fatalf("unexpected error marshalling InitConfiguration: %v", err)
	}
	t.Logf("\n%s", bytes)

	obj, err := UnmarshalFromYamlForCodecs(bytes, kubeadmapiv1beta1.SchemeGroupVersion, codecs)
	if err != nil {
		t.Fatalf("unexpected error unmarshalling InitConfiguration: %v", err)
	}

	cfg2, ok := obj.(*kubeadmapiv1beta1.InitConfiguration)
	if !ok || cfg2 == nil {
		t.Fatal("did not get InitConfiguration back")
	}
	if !reflect.DeepEqual(*cfg, *cfg2) {
		t.Errorf("expected %v, got %v", *cfg, *cfg2)
	}
}

// {{InitConfiguration kubeadm.k8s.io/v1beta1} [{<nil>  nil <nil> [] []}] {testNode /var/run/cri.sock [] map[]} {10.100.0.1  4332} {0xc4200ad2c0 <nil>} {10.100.0.0/24 10.100.1.0/24 cluster.local} stable-1.11 map[] map[] map[] [] [] [] [] /etc/kubernetes/pki k8s.gcr.io  { /var/log/kubernetes/audit 0x156e2f4} map[] kubernetes}
// {{InitConfiguration kubeadm.k8s.io/v1beta1} [{<nil>  &Duration{Duration:24h0m0s,} <nil> [signing authentication] [system:bootstrappers:kubeadm:default-node-token]}] {testNode /var/run/cri.sock [] map[]} {10.100.0.1  4332} {0xc4205c5260 <nil>} {10.100.0.0/24 10.100.1.0/24 cluster.local} stable-1.11 map[] map[] map[] [] [] [] [] /etc/kubernetes/pki k8s.gcr.io  { /var/log/kubernetes/audit 0xc4204dd82c} map[] kubernetes}

// {{InitConfiguration kubeadm.k8s.io/v1beta1} [{abcdef.abcdef0123456789  nil <nil> [] []}] {testNode /var/run/cri.sock [] map[]} {10.100.0.1  4332} {0xc42012ca80 <nil>} {10.100.0.0/24 10.100.1.0/24 cluster.local} stable-1.11 map[] map[] map[] [] [] [] [] /etc/kubernetes/pki k8s.gcr.io  { /var/log/kubernetes/audit 0x156e2f4} map[] kubernetes}
// {{InitConfiguration kubeadm.k8s.io/v1beta1} [{abcdef.abcdef0123456789  &Duration{Duration:24h0m0s,} <nil> [signing authentication] [system:bootstrappers:kubeadm:default-node-token]}] {testNode /var/run/cri.sock [] map[]} {10.100.0.1  4332} {0xc42039d1a0 <nil>} {10.100.0.0/24 10.100.1.0/24 cluster.local} stable-1.11 map[] map[] map[] [] [] [] [] /etc/kubernetes/pki k8s.gcr.io  { /var/log/kubernetes/audit 0xc4204fef3c} map[] kubernetes}

func TestSplitYAMLDocuments(t *testing.T) {
	var tests = []struct {
		name         string
		fileContents []byte
		gvkmap       map[schema.GroupVersionKind][]byte
		expectedErr  bool
	}{
		{
			name:         "FooOnly",
			fileContents: files["foo"],
			gvkmap: map[schema.GroupVersionKind][]byte{
				{Group: "foo.k8s.io", Version: "v1", Kind: "Foo"}: files["foo"],
			},
		},
		{
			name:         "FooBar",
			fileContents: bytes.Join([][]byte{files["foo"], files["bar"]}, []byte(constants.YAMLDocumentSeparator)),
			gvkmap: map[schema.GroupVersionKind][]byte{
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

			gvkmap, err := SplitYAMLDocuments(rt.fileContents)
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
