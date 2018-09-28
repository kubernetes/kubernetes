/*
Copyright 2018 The Kubernetes Authors.

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

package config

import (
	"bytes"
	"testing"

	kubeadmapiv1alpha3 "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/v1alpha3"
	"k8s.io/kubernetes/cmd/kubeadm/app/constants"
)

var files = map[string][]byte{
	"Master_v1alpha1": []byte(`
apiVersion: kubeadm.k8s.io/v1alpha1
kind: InitConfiguration
`),
	"Node_v1alpha1": []byte(`
apiVersion: kubeadm.k8s.io/v1alpha1
kind: NodeConfiguration
`),
	"Master_v1alpha2": []byte(`
apiVersion: kubeadm.k8s.io/v1alpha2
kind: MasterConfiguration
`),
	"Node_v1alpha2": []byte(`
apiVersion: kubeadm.k8s.io/v1alpha2
kind: NodeConfiguration
`),
	"Init_v1alpha3": []byte(`
apiVersion: kubeadm.k8s.io/v1alpha3
kind: InitConfiguration
`),
	"Join_v1alpha3": []byte(`
apiVersion: kubeadm.k8s.io/v1alpha3
kind: JoinConfiguration
`),
	"NoKind": []byte(`
apiVersion: baz.k8s.io/v1
foo: foo
bar: bar
`),
	"NoAPIVersion": []byte(`
kind: Bar
foo: foo
bar: bar
`),
	"Foo": []byte(`
apiVersion: foo.k8s.io/v1
kind: Foo
`),
}

func TestDetectUnsupportedVersion(t *testing.T) {
	var tests = []struct {
		name         string
		fileContents []byte
		expectedErr  bool
	}{
		{
			name:         "Master_v1alpha1",
			fileContents: files["Master_v1alpha1"],
			expectedErr:  true,
		},
		{
			name:         "Node_v1alpha1",
			fileContents: files["Node_v1alpha1"],
			expectedErr:  true,
		},
		{
			name:         "Master_v1alpha2",
			fileContents: files["Master_v1alpha2"],
			expectedErr:  true,
		},
		{
			name:         "Node_v1alpha2",
			fileContents: files["Node_v1alpha2"],
			expectedErr:  true,
		},
		{
			name:         "Init_v1alpha3",
			fileContents: files["Init_v1alpha3"],
		},
		{
			name:         "Join_v1alpha3",
			fileContents: files["Join_v1alpha3"],
		},
		{
			name:         "DuplicateInit",
			fileContents: bytes.Join([][]byte{files["Init_v1alpha3"], files["Init_v1alpha3"]}, []byte(constants.YAMLDocumentSeparator)),
			expectedErr:  true,
		},
		{
			name:         "DuplicateJoin",
			fileContents: bytes.Join([][]byte{files["Join_v1alpha3"], files["Join_v1alpha3"]}, []byte(constants.YAMLDocumentSeparator)),
			expectedErr:  true,
		},
		{
			name:         "NoKind",
			fileContents: files["NoKind"],
			expectedErr:  true,
		},
		{
			name:         "NoAPIVersion",
			fileContents: files["NoAPIVersion"],
			expectedErr:  true,
		},
		{
			name:         "v1alpha1InMultiple",
			fileContents: bytes.Join([][]byte{files["Foo"], files["Master_v1alpha1"]}, []byte(constants.YAMLDocumentSeparator)),
			expectedErr:  true,
		},
		// TODO: implement mustnotMix v1alpha3 v1beta1 after introducing v1beta1
		{
			name:         "MustNotMixInitJoin",
			fileContents: bytes.Join([][]byte{files["Init_v1alpha3"], files["Join_v1alpha3"]}, []byte(constants.YAMLDocumentSeparator)),
			expectedErr:  true,
		},
	}

	for _, rt := range tests {
		t.Run(rt.name, func(t2 *testing.T) {

			err := DetectUnsupportedVersion(rt.fileContents)
			if (err != nil) != rt.expectedErr {
				t2.Errorf("expected error: %t, actual: %t", rt.expectedErr, err != nil)
			}
		})
	}
}

func TestLowercaseSANs(t *testing.T) {
	tests := []struct {
		name string
		in   []string
		out  []string
	}{
		{
			name: "empty struct",
		},
		{
			name: "already lowercase",
			in:   []string{"example.k8s.io"},
			out:  []string{"example.k8s.io"},
		},
		{
			name: "ip addresses and uppercase",
			in:   []string{"EXAMPLE.k8s.io", "10.100.0.1"},
			out:  []string{"example.k8s.io", "10.100.0.1"},
		},
		{
			name: "punycode and uppercase",
			in:   []string{"xn--7gq663byk9a.xn--fiqz9s", "ANOTHEREXAMPLE.k8s.io"},
			out:  []string{"xn--7gq663byk9a.xn--fiqz9s", "anotherexample.k8s.io"},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			cfg := &kubeadmapiv1alpha3.InitConfiguration{
				ClusterConfiguration: kubeadmapiv1alpha3.ClusterConfiguration{
					APIServerCertSANs: test.in,
				},
			}

			LowercaseSANs(cfg.APIServerCertSANs)

			if len(cfg.APIServerCertSANs) != len(test.out) {
				t.Fatalf("expected %d elements, got %d", len(test.out), len(cfg.APIServerCertSANs))
			}

			for i, expected := range test.out {
				if cfg.APIServerCertSANs[i] != expected {
					t.Errorf("expected element %d to be %q, got %q", i, expected, cfg.APIServerCertSANs[i])
				}
			}
		})
	}
}

func TestVerifyAPIServerBindAddress(t *testing.T) {
	tests := []struct {
		name          string
		address       string
		expectedError bool
	}{
		{
			name:    "valid address: IPV4",
			address: "192.168.0.1",
		},
		{
			name:    "valid address: IPV6",
			address: "2001:db8:85a3::8a2e:370:7334",
		},
		{
			name:          "invalid address: not a global unicast 0.0.0.0",
			address:       "0.0.0.0",
			expectedError: true,
		},
		{
			name:          "invalid address: not a global unicast 127.0.0.1",
			address:       "127.0.0.1",
			expectedError: true,
		},
		{
			name:          "invalid address: not a global unicast ::",
			address:       "::",
			expectedError: true,
		},
		{
			name:          "invalid address: cannot parse IPV4",
			address:       "255.255.255.255.255",
			expectedError: true,
		},
		{
			name:          "invalid address: cannot parse IPV6",
			address:       "2a00:800::2a00:800:10102a00",
			expectedError: true,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if err := VerifyAPIServerBindAddress(test.address); (err != nil) != test.expectedError {
				t.Errorf("expected error: %v, got %v, error: %v", test.expectedError, (err != nil), err)
			}
		})
	}
}
