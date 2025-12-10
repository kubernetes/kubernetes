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
	"fmt"
	"os"
	"path/filepath"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	"github.com/lithammer/dedent"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/utils/ptr"

	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeadmapiv1 "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/v1beta4"
	"k8s.io/kubernetes/cmd/kubeadm/app/constants"
)

func TestLoadInitConfigurationFromFile(t *testing.T) {
	tmpdir, err := os.MkdirTemp("", "")
	if err != nil {
		t.Fatalf("Couldn't create tmpdir: %v", err)
	}
	defer func() {
		if err := os.RemoveAll(tmpdir); err != nil {
			t.Fatalf("Couldn't remove tmpdir: %v", err)
		}
	}()
	filename := "kubeadmConfig"
	filePath := filepath.Join(tmpdir, filename)
	options := LoadOrDefaultConfigurationOptions{}

	tests := []struct {
		name         string
		cfgPath      string
		fileContents string
		wantErr      bool
	}{
		{
			name:    "Config file does not exists",
			cfgPath: "tmp",
			wantErr: true,
		},
		{
			name:    "Valid kubeadm config",
			cfgPath: filePath,
			fileContents: dedent.Dedent(`
				apiVersion: kubeadm.k8s.io/v1beta4
				kind: InitConfiguration
		`),
			wantErr: false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.cfgPath == filePath {
				err = os.WriteFile(tt.cfgPath, []byte(tt.fileContents), 0644)
				if err != nil {
					t.Fatalf("Couldn't write content to file: %v", err)
				}
				defer func() {
					if err := os.RemoveAll(filePath); err != nil {
						t.Fatalf("Couldn't remove filePath: %v", err)
					}
				}()
			}

			_, err = LoadInitConfigurationFromFile(tt.cfgPath, options)
			if (err != nil) != tt.wantErr {
				t.Errorf("LoadInitConfigurationFromFile() error = %v, wantErr %v", err, tt.wantErr)
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
		for _, format := range formats {
			t.Run(fmt.Sprintf("%s_%s", tc.desc, format.name), func(t *testing.T) {
				b, err := format.marshal(tc.cfg)
				if err != nil {
					t.Fatalf("unexpected error while marshalling to %s: %v", format.name, err)
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
}

func TestBytesToInitConfiguration(t *testing.T) {
	tests := []struct {
		name          string
		cfg           interface{}
		expectedCfg   kubeadmapi.InitConfiguration
		expectedError bool
		skipCRIDetect bool
	}{
		{
			name: "default config is set correctly",
			cfg: kubeadmapiv1.InitConfiguration{
				TypeMeta: metav1.TypeMeta{
					APIVersion: kubeadmapiv1.SchemeGroupVersion.String(),
					Kind:       constants.InitConfigurationKind,
				},
			},
			expectedCfg: kubeadmapi.InitConfiguration{
				LocalAPIEndpoint: kubeadmapi.APIEndpoint{
					AdvertiseAddress: "",
					BindPort:         0,
				},
				NodeRegistration: kubeadmapi.NodeRegistrationOptions{
					CRISocket: "unix:///var/run/containerd/containerd.sock",
					Name:      "",
					Taints: []v1.Taint{
						{
							Key:    "node-role.kubernetes.io/control-plane",
							Effect: "NoSchedule",
						},
					},
					ImagePullPolicy: "IfNotPresent",
					ImagePullSerial: ptr.To(true),
				},
				ClusterConfiguration: kubeadmapi.ClusterConfiguration{
					Etcd: kubeadmapi.Etcd{
						Local: &kubeadmapi.LocalEtcd{
							DataDir: "/var/lib/etcd",
						},
					},
					KubernetesVersion:   "stable-1",
					ImageRepository:     kubeadmapiv1.DefaultImageRepository,
					ClusterName:         kubeadmapiv1.DefaultClusterName,
					EncryptionAlgorithm: kubeadmapi.EncryptionAlgorithmType(kubeadmapiv1.DefaultEncryptionAlgorithm),
					Networking: kubeadmapi.Networking{
						ServiceSubnet: "10.96.0.0/12",
						DNSDomain:     "cluster.local",
					},
					CertificatesDir: "/etc/kubernetes/pki",
				},
			},
			skipCRIDetect: true,
		},
		{
			name: "partial config with custom values",
			cfg: kubeadmapiv1.InitConfiguration{
				TypeMeta: metav1.TypeMeta{
					APIVersion: kubeadmapiv1.SchemeGroupVersion.String(),
					Kind:       constants.InitConfigurationKind,
				},
				NodeRegistration: kubeadmapiv1.NodeRegistrationOptions{
					Name:      "test-node",
					CRISocket: "unix:///var/run/containerd/containerd.sock",
				},
			},
			expectedCfg: kubeadmapi.InitConfiguration{
				LocalAPIEndpoint: kubeadmapi.APIEndpoint{
					AdvertiseAddress: "",
					BindPort:         0,
				},
				NodeRegistration: kubeadmapi.NodeRegistrationOptions{
					CRISocket: "unix:///var/run/containerd/containerd.sock",
					Name:      "test-node",
					Taints: []v1.Taint{
						{
							Key:    "node-role.kubernetes.io/control-plane",
							Effect: "NoSchedule",
						},
					},
					ImagePullPolicy: "IfNotPresent",
					ImagePullSerial: ptr.To(true),
				},
				ClusterConfiguration: kubeadmapi.ClusterConfiguration{
					Etcd: kubeadmapi.Etcd{
						Local: &kubeadmapi.LocalEtcd{
							DataDir: "/var/lib/etcd",
						},
					},
					KubernetesVersion:   "stable-1",
					ImageRepository:     kubeadmapiv1.DefaultImageRepository,
					ClusterName:         kubeadmapiv1.DefaultClusterName,
					EncryptionAlgorithm: kubeadmapi.EncryptionAlgorithmType(kubeadmapiv1.DefaultEncryptionAlgorithm),
					Networking: kubeadmapi.Networking{
						ServiceSubnet: "10.96.0.0/12",
						DNSDomain:     "cluster.local",
					},
					CertificatesDir: "/etc/kubernetes/pki",
				},
			},
			skipCRIDetect: true,
		},
		{
			name: "invalid configuration type",
			cfg: kubeadmapiv1.UpgradeConfiguration{
				TypeMeta: metav1.TypeMeta{
					APIVersion: kubeadmapiv1.SchemeGroupVersion.String(),
					Kind:       constants.UpgradeConfigurationKind,
				},
			},
			expectedError: true,
			skipCRIDetect: true,
		},
	}

	for _, tc := range tests {
		for _, format := range formats {
			t.Run(fmt.Sprintf("%s_%s", tc.name, format.name), func(t *testing.T) {
				b, err := format.marshal(tc.cfg)
				if err != nil {
					t.Fatalf("unexpected error marshaling %s: %v", format.name, err)
				}

				cfg, err := BytesToInitConfiguration(b, tc.skipCRIDetect)
				if (err != nil) != tc.expectedError {
					t.Fatalf("expected error: %v, got error: %v\nError: %v", tc.expectedError, err != nil, err)
				}

				if !tc.expectedError {
					// Ignore dynamic fields that may be set during defaulting
					diffOpts := []cmp.Option{
						cmpopts.IgnoreFields(kubeadmapi.NodeRegistrationOptions{}, "Name"),
						cmpopts.IgnoreFields(kubeadmapi.InitConfiguration{}, "Timeouts", "BootstrapTokens", "LocalAPIEndpoint"),
						cmpopts.IgnoreFields(kubeadmapi.APIServer{}, "TimeoutForControlPlane"),
						cmpopts.IgnoreFields(kubeadmapi.ClusterConfiguration{}, "ComponentConfigs", "KubernetesVersion",
							"CertificateValidityPeriod", "CACertificateValidityPeriod"),
					}

					if diff := cmp.Diff(*cfg, tc.expectedCfg, diffOpts...); diff != "" {
						t.Fatalf("unexpected configuration difference (-want +got):\n%s", diff)
					}
				}
			})
		}
	}
}
