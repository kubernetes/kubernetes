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
	"fmt"
	"os"
	"path/filepath"
	"testing"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"sigs.k8s.io/yaml"

	"k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeadmapiv1 "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/v1beta4"
	"k8s.io/kubernetes/cmd/kubeadm/app/constants"
)

func TestLoadInitConfigurationFromFile(t *testing.T) {
	certDir := "/tmp/foo"
	clusterCfg := []byte(fmt.Sprintf(`
apiVersion: %s
kind: ClusterConfiguration
certificatesDir: %s
kubernetesVersion: %s`, kubeadmapiv1.SchemeGroupVersion.String(), certDir, constants.MinimumControlPlaneVersion.String()))

	// Create temp folder for the test case
	tmpdir, err := os.MkdirTemp("", "")
	if err != nil {
		t.Fatalf("Couldn't create tmpdir: %v", err)
	}
	defer os.RemoveAll(tmpdir)

	// cfgFiles is in cluster_test.go
	var tests = []struct {
		name         string
		fileContents []byte
		expectErr    bool
		validate     func(*testing.T, *kubeadm.InitConfiguration)
	}{
		{
			name:         "v1beta3.partial1",
			fileContents: cfgFiles["InitConfiguration_v1beta3"],
		},
		{
			name: "v1beta3.partial2",
			fileContents: bytes.Join([][]byte{
				cfgFiles["InitConfiguration_v1beta3"],
				clusterCfg,
			}, []byte(constants.YAMLDocumentSeparator)),
			validate: func(t *testing.T, cfg *kubeadm.InitConfiguration) {
				if cfg.ClusterConfiguration.CertificatesDir != certDir {
					t.Errorf("CertificatesDir from ClusterConfiguration holds the wrong value, Expected: %v. Actual: %v", certDir, cfg.ClusterConfiguration.CertificatesDir)
				}
			},
		},
		{
			name: "v1beta3.full",
			fileContents: bytes.Join([][]byte{
				cfgFiles["InitConfiguration_v1beta3"],
				cfgFiles["ClusterConfiguration_v1beta3"],
				cfgFiles["Kube-proxy_componentconfig"],
				cfgFiles["Kubelet_componentconfig"],
			}, []byte(constants.YAMLDocumentSeparator)),
		},
		{
			name: "v1beta4.full",
			fileContents: bytes.Join([][]byte{
				cfgFiles["InitConfiguration_v1beta4"],
				cfgFiles["ClusterConfiguration_v1beta4"],
				cfgFiles["Kube-proxy_componentconfig"],
				cfgFiles["Kubelet_componentconfig"],
			}, []byte(constants.YAMLDocumentSeparator)),
		},
	}

	for _, rt := range tests {
		t.Run(rt.name, func(t2 *testing.T) {
			cfgPath := filepath.Join(tmpdir, rt.name)
			err := os.WriteFile(cfgPath, rt.fileContents, 0644)
			if err != nil {
				t.Errorf("Couldn't create file: %v", err)
				return
			}

			opts := LoadOrDefaultConfigurationOptions{
				SkipCRIDetect: true,
			}

			obj, err := LoadInitConfigurationFromFile(cfgPath, opts)
			if rt.expectErr {
				if err == nil {
					t.Error("Unexpected success")
				}
			} else {
				if err != nil {
					t.Errorf("Error reading file: %v", err)
					return
				}

				if obj == nil {
					t.Error("Unexpected nil return value")
				}
			}
			// exec additional validation on the returned value
			if rt.validate != nil {
				rt.validate(t, obj)
			}
		})
	}
}

func TestDefaultTaintsMarshaling(t *testing.T) {
	tests := []struct {
		desc             string
		cfg              kubeadmapiv1.InitConfiguration
		expectedTaintCnt int
	}{
		{
			desc: "Uninitialized nodeRegistration field produces expected taints",
			cfg: kubeadmapiv1.InitConfiguration{
				TypeMeta: metav1.TypeMeta{
					APIVersion: kubeadmapiv1.SchemeGroupVersion.String(),
					Kind:       constants.InitConfigurationKind,
				},
			},
			expectedTaintCnt: 1,
		},
		{
			desc: "Uninitialized taints field produces expected taints",
			cfg: kubeadmapiv1.InitConfiguration{
				TypeMeta: metav1.TypeMeta{
					APIVersion: kubeadmapiv1.SchemeGroupVersion.String(),
					Kind:       constants.InitConfigurationKind,
				},
			},
			expectedTaintCnt: 1,
		},
		{
			desc: "Forsing taints to an empty slice produces no taints",
			cfg: kubeadmapiv1.InitConfiguration{
				TypeMeta: metav1.TypeMeta{
					APIVersion: kubeadmapiv1.SchemeGroupVersion.String(),
					Kind:       constants.InitConfigurationKind,
				},
				NodeRegistration: kubeadmapiv1.NodeRegistrationOptions{
					Taints: []v1.Taint{},
				},
			},
			expectedTaintCnt: 0,
		},
		{
			desc: "Custom taints are used",
			cfg: kubeadmapiv1.InitConfiguration{
				TypeMeta: metav1.TypeMeta{
					APIVersion: kubeadmapiv1.SchemeGroupVersion.String(),
					Kind:       constants.InitConfigurationKind,
				},
				NodeRegistration: kubeadmapiv1.NodeRegistrationOptions{
					Taints: []v1.Taint{
						{Key: "taint1"},
						{Key: "taint2"},
					},
				},
			},
			expectedTaintCnt: 2,
		},
	}

	for _, tc := range tests {
		t.Run(tc.desc, func(t *testing.T) {
			b, err := yaml.Marshal(tc.cfg)
			if err != nil {
				t.Fatalf("unexpected error while marshalling to YAML: %v", err)
			}

			cfg, err := BytesToInitConfiguration(b, true)
			if err != nil {
				t.Fatalf("unexpected error of BytesToInitConfiguration: %v\nconfig: %s", err, string(b))
			}

			if tc.expectedTaintCnt != len(cfg.NodeRegistration.Taints) {
				t.Fatalf("unexpected taints count\nexpected: %d\ngot: %d\ntaints: %v", tc.expectedTaintCnt, len(cfg.NodeRegistration.Taints), cfg.NodeRegistration.Taints)
			}
		})
	}
}
