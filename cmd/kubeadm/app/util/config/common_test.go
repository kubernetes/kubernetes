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
	"io/ioutil"
	"os"
	"testing"

	"github.com/renstrom/dedent"

	kubeadmapiv1beta1 "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/v1beta1"
	"k8s.io/kubernetes/cmd/kubeadm/app/constants"
	kubeadmutil "k8s.io/kubernetes/cmd/kubeadm/app/util"
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
	"Init_v1beta1": []byte(`
apiVersion: kubeadm.k8s.io/v1beta1
kind: InitConfiguration
`),
	"Join_v1beta1": []byte(`
apiVersion: kubeadm.k8s.io/v1beta1
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
			name:         "Init_v1beta1",
			fileContents: files["Init_v1beta1"],
		},
		{
			name:         "Join_v1beta1",
			fileContents: files["Join_v1beta1"],
		},
		{
			name:         "DuplicateInit v1alpha3",
			fileContents: bytes.Join([][]byte{files["Init_v1alpha3"], files["Init_v1alpha3"]}, []byte(constants.YAMLDocumentSeparator)),
			expectedErr:  true,
		},
		{
			name:         "DuplicateInit v1beta1",
			fileContents: bytes.Join([][]byte{files["Init_v1beta1"], files["Init_v1beta1"]}, []byte(constants.YAMLDocumentSeparator)),
			expectedErr:  true,
		},
		{
			name:         "DuplicateInit v1beta1 and v1alpha3",
			fileContents: bytes.Join([][]byte{files["Init_v1beta1"], files["Init_v1alpha3"]}, []byte(constants.YAMLDocumentSeparator)),
			expectedErr:  true,
		},
		{
			name:         "DuplicateJoin v1alpha3",
			fileContents: bytes.Join([][]byte{files["Join_v1alpha3"], files["Join_v1alpha3"]}, []byte(constants.YAMLDocumentSeparator)),
			expectedErr:  true,
		},
		{
			name:         "DuplicateJoin v1beta1",
			fileContents: bytes.Join([][]byte{files["Join_v1beta1"], files["Join_v1beta1"]}, []byte(constants.YAMLDocumentSeparator)),
			expectedErr:  true,
		},
		{
			name:         "DuplicateJoin v1beta1 and v1alpha3",
			fileContents: bytes.Join([][]byte{files["Join_v1beta1"], files["Join_v1alpha3"]}, []byte(constants.YAMLDocumentSeparator)),
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
			name:         "Ignore other Kind",
			fileContents: bytes.Join([][]byte{files["Foo"], files["Master_v1alpha3"]}, []byte(constants.YAMLDocumentSeparator)),
		},
		{
			name:         "Ignore other Kind",
			fileContents: bytes.Join([][]byte{files["Foo"], files["Master_v1beta1"]}, []byte(constants.YAMLDocumentSeparator)),
		},
		// CanMixInitJoin cases used to be MustNotMixInitJoin, however due to UX issues DetectUnsupportedVersion had to tolerate that.
		// So the following tests actually verify, that Init and Join can be mixed together with no error.
		{
			name:         "CanMixInitJoin v1alpha3",
			fileContents: bytes.Join([][]byte{files["Init_v1alpha3"], files["Join_v1alpha3"]}, []byte(constants.YAMLDocumentSeparator)),
			expectedErr:  false,
		},
		{
			name:         "CanMixInitJoin v1alpha3 - v1beta1",
			fileContents: bytes.Join([][]byte{files["Init_v1alpha3"], files["Join_v1beta1"]}, []byte(constants.YAMLDocumentSeparator)),
			expectedErr:  false,
		},
		{
			name:         "CanMixInitJoin v1beta1 - v1alpha3",
			fileContents: bytes.Join([][]byte{files["Init_v1beta1"], files["Join_v1alpha3"]}, []byte(constants.YAMLDocumentSeparator)),
			expectedErr:  false,
		},
		{
			name:         "CanMixInitJoin v1beta1",
			fileContents: bytes.Join([][]byte{files["Init_v1beta1"], files["Join_v1beta1"]}, []byte(constants.YAMLDocumentSeparator)),
			expectedErr:  false,
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
			cfg := &kubeadmapiv1beta1.InitConfiguration{
				ClusterConfiguration: kubeadmapiv1beta1.ClusterConfiguration{
					APIServer: kubeadmapiv1beta1.APIServer{
						CertSANs: test.in,
					},
				},
			}

			LowercaseSANs(cfg.APIServer.CertSANs)

			if len(cfg.APIServer.CertSANs) != len(test.out) {
				t.Fatalf("expected %d elements, got %d", len(test.out), len(cfg.APIServer.CertSANs))
			}

			for i, expected := range test.out {
				if cfg.APIServer.CertSANs[i] != expected {
					t.Errorf("expected element %d to be %q, got %q", i, expected, cfg.APIServer.CertSANs[i])
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

func TestMigrateOldConfigFromFile(t *testing.T) {
	tests := []struct {
		desc          string
		oldCfg        string
		expectedKinds []string
		expectErr     bool
	}{
		{
			desc:      "empty file produces empty result",
			oldCfg:    "",
			expectErr: false,
		},
		{
			desc: "bad config produces error",
			oldCfg: dedent.Dedent(`
			apiVersion: kubeadm.k8s.io/v1alpha3
			`),
			expectErr: true,
		},
		{
			desc: "InitConfiguration only gets migrated",
			oldCfg: dedent.Dedent(`
			apiVersion: kubeadm.k8s.io/v1alpha3
			kind: InitConfiguration
			`),
			expectedKinds: []string{
				constants.InitConfigurationKind,
				constants.ClusterConfigurationKind,
			},
			expectErr: false,
		},
		{
			desc: "ClusterConfiguration only gets migrated",
			oldCfg: dedent.Dedent(`
			apiVersion: kubeadm.k8s.io/v1alpha3
			kind: ClusterConfiguration
			`),
			expectedKinds: []string{
				constants.InitConfigurationKind,
				constants.ClusterConfigurationKind,
			},
			expectErr: false,
		},
		{
			desc: "JoinConfiguration only gets migrated",
			oldCfg: dedent.Dedent(`
			apiVersion: kubeadm.k8s.io/v1alpha3
			kind: JoinConfiguration
			token: abcdef.0123456789abcdef
			discoveryTokenAPIServers:
			- kube-apiserver:6443
			discoveryTokenUnsafeSkipCAVerification: true
			`),
			expectedKinds: []string{
				constants.JoinConfigurationKind,
			},
			expectErr: false,
		},
		{
			desc: "Init + Cluster Configurations are migrated",
			oldCfg: dedent.Dedent(`
			apiVersion: kubeadm.k8s.io/v1alpha3
			kind: InitConfiguration
			---
			apiVersion: kubeadm.k8s.io/v1alpha3
			kind: ClusterConfiguration
			`),
			expectedKinds: []string{
				constants.InitConfigurationKind,
				constants.ClusterConfigurationKind,
			},
			expectErr: false,
		},
		{
			desc: "Init + Join Configurations are migrated",
			oldCfg: dedent.Dedent(`
			apiVersion: kubeadm.k8s.io/v1alpha3
			kind: InitConfiguration
			---
			apiVersion: kubeadm.k8s.io/v1alpha3
			kind: JoinConfiguration
			token: abcdef.0123456789abcdef
			discoveryTokenAPIServers:
			- kube-apiserver:6443
			discoveryTokenUnsafeSkipCAVerification: true
			`),
			expectedKinds: []string{
				constants.InitConfigurationKind,
				constants.ClusterConfigurationKind,
				constants.JoinConfigurationKind,
			},
			expectErr: false,
		},
		{
			desc: "Cluster + Join Configurations are migrated",
			oldCfg: dedent.Dedent(`
			apiVersion: kubeadm.k8s.io/v1alpha3
			kind: ClusterConfiguration
			---
			apiVersion: kubeadm.k8s.io/v1alpha3
			kind: JoinConfiguration
			token: abcdef.0123456789abcdef
			discoveryTokenAPIServers:
			- kube-apiserver:6443
			discoveryTokenUnsafeSkipCAVerification: true
			`),
			expectedKinds: []string{
				constants.InitConfigurationKind,
				constants.ClusterConfigurationKind,
				constants.JoinConfigurationKind,
			},
			expectErr: false,
		},
		{
			desc: "Init + Cluster + Join Configurations are migrated",
			oldCfg: dedent.Dedent(`
			apiVersion: kubeadm.k8s.io/v1alpha3
			kind: InitConfiguration
			---
			apiVersion: kubeadm.k8s.io/v1alpha3
			kind: ClusterConfiguration
			---
			apiVersion: kubeadm.k8s.io/v1alpha3
			kind: JoinConfiguration
			token: abcdef.0123456789abcdef
			discoveryTokenAPIServers:
			- kube-apiserver:6443
			discoveryTokenUnsafeSkipCAVerification: true
			`),
			expectedKinds: []string{
				constants.InitConfigurationKind,
				constants.ClusterConfigurationKind,
				constants.JoinConfigurationKind,
			},
			expectErr: false,
		},
		{
			desc: "component configs are not migrated",
			oldCfg: dedent.Dedent(`
			apiVersion: kubeadm.k8s.io/v1alpha3
			kind: InitConfiguration
			---
			apiVersion: kubeadm.k8s.io/v1alpha3
			kind: ClusterConfiguration
			---
			apiVersion: kubeadm.k8s.io/v1alpha3
			kind: JoinConfiguration
			token: abcdef.0123456789abcdef
			discoveryTokenAPIServers:
			- kube-apiserver:6443
			discoveryTokenUnsafeSkipCAVerification: true
			---
			apiVersion: kubeproxy.config.k8s.io/v1alpha1
			kind: KubeProxyConfiguration
			---
			apiVersion: kubelet.config.k8s.io/v1beta1
			kind: KubeletConfiguration
			`),
			expectedKinds: []string{
				constants.InitConfigurationKind,
				constants.ClusterConfigurationKind,
				constants.JoinConfigurationKind,
			},
			expectErr: false,
		},
	}

	for _, test := range tests {
		t.Run(test.desc, func(t *testing.T) {
			file, err := ioutil.TempFile("", "")
			if err != nil {
				t.Fatalf("could not create temporary test file: %v", err)
			}
			fileName := file.Name()
			defer os.Remove(fileName)

			_, err = file.WriteString(test.oldCfg)
			file.Close()
			if err != nil {
				t.Fatalf("could not write contents of old config: %v", err)
			}

			b, err := MigrateOldConfigFromFile(fileName)
			if test.expectErr {
				if err == nil {
					t.Fatalf("unexpected success:\n%s", b)
				}
			} else {
				if err != nil {
					t.Fatalf("unexpected failure: %v", err)
				}
				gvks, err := kubeadmutil.GroupVersionKindsFromBytes(b)
				if err != nil {
					t.Fatalf("unexpected error returned by GroupVersionKindsFromBytes: %v", err)
				}
				if len(gvks) != len(test.expectedKinds) {
					t.Fatalf("length mismatch between resulting gvks and expected kinds:\n\tlen(gvks)=%d\n\tlen(expectedKinds)=%d",
						len(gvks), len(test.expectedKinds))
				}
				for _, expectedKind := range test.expectedKinds {
					if !kubeadmutil.GroupVersionKindsHasKind(gvks, expectedKind) {
						t.Fatalf("migration failed to produce config kind: %s", expectedKind)
					}
				}
			}
		})
	}
}
