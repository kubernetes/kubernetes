/*
Copyright 2024 The Kubernetes Authors.

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
	"k8s.io/utils/ptr"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeadmapiv1 "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/v1beta4"
	"k8s.io/kubernetes/cmd/kubeadm/app/constants"
)

func TestBytesToUpgradeConfiguration(t *testing.T) {
	tests := []struct {
		name          string
		cfg           interface{}
		expectedCfg   kubeadmapi.UpgradeConfiguration
		expectedError bool
	}{
		{
			name: "default config is set correctly",
			cfg: kubeadmapiv1.UpgradeConfiguration{
				TypeMeta: metav1.TypeMeta{
					APIVersion: kubeadmapiv1.SchemeGroupVersion.String(),
					Kind:       constants.UpgradeConfigurationKind,
				},
			},
			expectedCfg: kubeadmapi.UpgradeConfiguration{
				Apply: kubeadmapi.UpgradeApplyConfiguration{
					CertificateRenewal: ptr.To(true),
					EtcdUpgrade:        ptr.To(true),
					ImagePullPolicy:    v1.PullIfNotPresent,
					ImagePullSerial:    ptr.To(true),
				},
				Node: kubeadmapi.UpgradeNodeConfiguration{
					CertificateRenewal: ptr.To(true),
					EtcdUpgrade:        ptr.To(true),
					ImagePullPolicy:    v1.PullIfNotPresent,
					ImagePullSerial:    ptr.To(true),
				},
				Plan: kubeadmapi.UpgradePlanConfiguration{
					EtcdUpgrade: ptr.To(true),
				},
			},
		},
		{
			name: "cfg has part of fields configured",
			cfg: kubeadmapiv1.UpgradeConfiguration{
				Apply: kubeadmapiv1.UpgradeApplyConfiguration{
					CertificateRenewal: ptr.To(false),
				},
				Node: kubeadmapiv1.UpgradeNodeConfiguration{
					EtcdUpgrade: ptr.To(false),
				},
				Plan: kubeadmapiv1.UpgradePlanConfiguration{
					EtcdUpgrade: ptr.To(false),
				},
				TypeMeta: metav1.TypeMeta{
					APIVersion: kubeadmapiv1.SchemeGroupVersion.String(),
					Kind:       constants.UpgradeConfigurationKind,
				},
			},
			expectedCfg: kubeadmapi.UpgradeConfiguration{
				Apply: kubeadmapi.UpgradeApplyConfiguration{
					CertificateRenewal: ptr.To(false),
					EtcdUpgrade:        ptr.To(true),
					ImagePullPolicy:    v1.PullIfNotPresent,
					ImagePullSerial:    ptr.To(true),
				},
				Node: kubeadmapi.UpgradeNodeConfiguration{
					CertificateRenewal: ptr.To(true),
					EtcdUpgrade:        ptr.To(false),
					ImagePullPolicy:    v1.PullIfNotPresent,
					ImagePullSerial:    ptr.To(true),
				},
				Plan: kubeadmapi.UpgradePlanConfiguration{
					EtcdUpgrade: ptr.To(false),
				},
			},
		},
		{
			name: "no UpgradeConfiguration found",
			cfg: kubeadmapiv1.InitConfiguration{
				TypeMeta: metav1.TypeMeta{
					APIVersion: kubeadmapiv1.SchemeGroupVersion.String(),
					Kind:       constants.InitConfigurationKind,
				},
			},
			expectedError: true,
		},
	}

	for _, tc := range tests {
		for _, format := range formats {
			t.Run(fmt.Sprintf("%s_%s", tc.name, format.name), func(t *testing.T) {
				b, err := format.marshal(tc.cfg)
				if err != nil {
					t.Fatalf("unexpected error while marshalling to %s: %v", format.name, err)
				}

				cfg, err := BytesToUpgradeConfiguration(b)
				if (err != nil) != tc.expectedError {
					t.Fatalf("failed BytesToUpgradeConfiguration:\n\texpected error: %t\n\t  actual error: %v", tc.expectedError, err)
				}

				if err == nil {
					if diff := cmp.Diff(*cfg, tc.expectedCfg, cmpopts.IgnoreFields(kubeadmapi.UpgradeConfiguration{}, "Timeouts")); diff != "" {
						t.Fatalf("BytesToUpgradeConfiguration returned unexpected diff (-want,+got):\n%s", diff)
					}
				}
			})
		}
	}
}

func TestLoadUpgradeConfigurationFromFile(t *testing.T) {
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
				kind: UpgradeConfiguration
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

			_, err = LoadUpgradeConfigurationFromFile(tt.cfgPath, options)
			if (err != nil) != tt.wantErr {
				t.Errorf("LoadUpgradeConfigurationFromFile() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func TestDefaultedUpgradeConfiguration(t *testing.T) {
	options := LoadOrDefaultConfigurationOptions{}
	tests := []struct {
		name string
		cfg  *kubeadmapiv1.UpgradeConfiguration
		want *kubeadmapi.UpgradeConfiguration
	}{
		{
			name: "config is empty",
			cfg:  &kubeadmapiv1.UpgradeConfiguration{},
			want: &kubeadmapi.UpgradeConfiguration{
				Apply: kubeadmapi.UpgradeApplyConfiguration{
					CertificateRenewal: ptr.To(true),
					EtcdUpgrade:        ptr.To(true),
					ImagePullPolicy:    v1.PullIfNotPresent,
					ImagePullSerial:    ptr.To(true),
				},
				Node: kubeadmapi.UpgradeNodeConfiguration{
					CertificateRenewal: ptr.To(true),
					EtcdUpgrade:        ptr.To(true),
					ImagePullPolicy:    v1.PullIfNotPresent,
					ImagePullSerial:    ptr.To(true),
				},
				Plan: kubeadmapi.UpgradePlanConfiguration{
					EtcdUpgrade: ptr.To(true),
				},
			},
		},
		{
			name: "config has some fields configured",
			cfg: &kubeadmapiv1.UpgradeConfiguration{
				Apply: kubeadmapiv1.UpgradeApplyConfiguration{
					CertificateRenewal: ptr.To(false),
					ImagePullPolicy:    v1.PullAlways,
					ImagePullSerial:    ptr.To(false),
				},
				Node: kubeadmapiv1.UpgradeNodeConfiguration{
					EtcdUpgrade:     ptr.To(false),
					ImagePullPolicy: v1.PullAlways,
					ImagePullSerial: ptr.To(false),
				},
				Plan: kubeadmapiv1.UpgradePlanConfiguration{
					EtcdUpgrade: ptr.To(false),
				},
				TypeMeta: metav1.TypeMeta{
					APIVersion: kubeadmapiv1.SchemeGroupVersion.String(),
					Kind:       constants.UpgradeConfigurationKind,
				},
			},
			want: &kubeadmapi.UpgradeConfiguration{
				Apply: kubeadmapi.UpgradeApplyConfiguration{
					CertificateRenewal: ptr.To(false),
					EtcdUpgrade:        ptr.To(true),
					ImagePullPolicy:    v1.PullAlways,
					ImagePullSerial:    ptr.To(false),
				},
				Node: kubeadmapi.UpgradeNodeConfiguration{
					CertificateRenewal: ptr.To(true),
					EtcdUpgrade:        ptr.To(false),
					ImagePullPolicy:    v1.PullAlways,
					ImagePullSerial:    ptr.To(false),
				},
				Plan: kubeadmapi.UpgradePlanConfiguration{
					EtcdUpgrade: ptr.To(false),
				},
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, _ := DefaultedUpgradeConfiguration(tt.cfg, options)
			if diff := cmp.Diff(got, tt.want, cmpopts.IgnoreFields(kubeadmapi.UpgradeConfiguration{}, "Timeouts")); diff != "" {
				t.Errorf("DefaultedUpgradeConfiguration returned unexpected diff (-want,+got):\n%s", diff)
			}
		})
	}
}

func TestLoadOrDefaultUpgradeConfiguration(t *testing.T) {
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
		name    string
		cfgPath string
		cfg     *kubeadmapiv1.UpgradeConfiguration
		want    *kubeadmapi.UpgradeConfiguration
	}{
		{
			name:    "cfgpPath is empty, the result should be obtained from cfg",
			cfgPath: "",
			cfg: &kubeadmapiv1.UpgradeConfiguration{
				Apply: kubeadmapiv1.UpgradeApplyConfiguration{
					CertificateRenewal: ptr.To(false),
				},
				Node: kubeadmapiv1.UpgradeNodeConfiguration{
					EtcdUpgrade: ptr.To(false),
				},
				Plan: kubeadmapiv1.UpgradePlanConfiguration{
					EtcdUpgrade: ptr.To(false),
				},
				TypeMeta: metav1.TypeMeta{
					APIVersion: kubeadmapiv1.SchemeGroupVersion.String(),
					Kind:       constants.UpgradeConfigurationKind,
				},
			},
			want: &kubeadmapi.UpgradeConfiguration{
				Apply: kubeadmapi.UpgradeApplyConfiguration{
					CertificateRenewal: ptr.To(false),
					EtcdUpgrade:        ptr.To(true),
					ImagePullPolicy:    v1.PullIfNotPresent,
					ImagePullSerial:    ptr.To(true),
				},
				Node: kubeadmapi.UpgradeNodeConfiguration{
					CertificateRenewal: ptr.To(true),
					EtcdUpgrade:        ptr.To(false),
					ImagePullPolicy:    v1.PullIfNotPresent,
					ImagePullSerial:    ptr.To(true),
				},
				Plan: kubeadmapi.UpgradePlanConfiguration{
					EtcdUpgrade: ptr.To(false),
				},
			},
		},
		{
			name:    "cfgpPath is not empty, the result should be obtained from the configuration file",
			cfgPath: filePath,
			cfg: &kubeadmapiv1.UpgradeConfiguration{
				Apply: kubeadmapiv1.UpgradeApplyConfiguration{
					CertificateRenewal: ptr.To(false),
					EtcdUpgrade:        ptr.To(false),
					ImagePullPolicy:    v1.PullNever,
					ImagePullSerial:    ptr.To(false),
				},
				Node: kubeadmapiv1.UpgradeNodeConfiguration{
					CertificateRenewal: ptr.To(false),
					EtcdUpgrade:        ptr.To(false),
					ImagePullPolicy:    v1.PullNever,
					ImagePullSerial:    ptr.To(false),
				},
				Plan: kubeadmapiv1.UpgradePlanConfiguration{
					EtcdUpgrade: ptr.To(false),
				},
				TypeMeta: metav1.TypeMeta{
					APIVersion: kubeadmapiv1.SchemeGroupVersion.String(),
					Kind:       constants.UpgradeConfigurationKind,
				},
			},
			want: &kubeadmapi.UpgradeConfiguration{
				Apply: kubeadmapi.UpgradeApplyConfiguration{
					CertificateRenewal: ptr.To(false),
					EtcdUpgrade:        ptr.To(false),
					ImagePullPolicy:    v1.PullNever,
					ImagePullSerial:    ptr.To(false),
				},
				Node: kubeadmapi.UpgradeNodeConfiguration{
					CertificateRenewal: ptr.To(false),
					EtcdUpgrade:        ptr.To(false),
					ImagePullPolicy:    v1.PullNever,
					ImagePullSerial:    ptr.To(false),
				},
				Plan: kubeadmapi.UpgradePlanConfiguration{
					EtcdUpgrade: ptr.To(false),
				},
			},
		},
	}

	for _, tt := range tests {
		for _, format := range formats {
			t.Run(fmt.Sprintf("%s_%s", tt.name, format.name), func(t *testing.T) {
				bytes, err := format.marshal(tt.cfg)
				if err != nil {
					t.Fatalf("Could not marshal test config: %v", err)
				}
				err = os.WriteFile(filePath, bytes, 0644)
				if err != nil {
					t.Fatalf("Couldn't write content to file: %v", err)
				}

				got, _ := LoadOrDefaultUpgradeConfiguration(tt.cfgPath, tt.cfg, options)
				if diff := cmp.Diff(got, tt.want, cmpopts.IgnoreFields(kubeadmapi.UpgradeConfiguration{}, "Timeouts")); diff != "" {
					t.Errorf("LoadOrDefaultUpgradeConfiguration returned unexpected diff (-want,+got):\n%s", diff)
				}
			})
		}
	}
}
