/*
Copyright 2023 The Kubernetes Authors.

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
	"github.com/stretchr/testify/assert"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeadmapiv1 "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/v1beta4"
	"k8s.io/kubernetes/cmd/kubeadm/app/constants"
)

func TestBytesToResetConfiguration(t *testing.T) {
	options := LoadOrDefaultConfigurationOptions{}

	tests := []struct {
		name string
		cfg  *kubeadmapiv1.ResetConfiguration
		want *kubeadmapi.ResetConfiguration
	}{
		{
			name: "Normal configuration",
			cfg: &kubeadmapiv1.ResetConfiguration{
				TypeMeta: metav1.TypeMeta{
					APIVersion: kubeadmapiv1.SchemeGroupVersion.String(),
					Kind:       constants.ResetConfigurationKind,
				},
				CertificatesDir: "/custom/certs",
				CleanupTmpDir:   true,
			},
			want: &kubeadmapi.ResetConfiguration{
				CertificatesDir: "/custom/certs",
				CleanupTmpDir:   true,
				SkipPhases:      nil,
				CRISocket:       "unix:///var/run/containerd/containerd.sock",
			},
		},
		{
			name: "Default configuration",
			cfg: &kubeadmapiv1.ResetConfiguration{
				TypeMeta: metav1.TypeMeta{
					APIVersion: kubeadmapiv1.SchemeGroupVersion.String(),
					Kind:       constants.ResetConfigurationKind,
				},
			},
			want: &kubeadmapi.ResetConfiguration{
				CertificatesDir: "/etc/kubernetes/pki",
				CleanupTmpDir:   false,
				SkipPhases:      nil,
				CRISocket:       "unix:///var/run/containerd/containerd.sock",
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
				got, _ := BytesToResetConfiguration(bytes, options)
				if diff := cmp.Diff(got, tt.want, cmpopts.IgnoreFields(kubeadmapi.ResetConfiguration{}, "Timeouts")); diff != "" {
					t.Errorf("LoadResetConfigurationFromFile returned unexpected diff (-want,+got):\n%s", diff)
				}
			})
		}
	}
}

func TestLoadResetConfigurationFromFile(t *testing.T) {
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
				kind: ResetConfiguration`),
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

			_, err = LoadResetConfigurationFromFile(tt.cfgPath, options)
			if (err != nil) != tt.wantErr {
				t.Errorf("LoadResetConfigurationFromFile() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func TestSetResetDynamicDefaults(t *testing.T) {
	type args struct {
		cfg           *kubeadmapi.ResetConfiguration
		skipCRIDetect bool
	}
	tests := []struct {
		name string
		args args
	}{
		{
			name: "CRISocket is empty and skipCRIDetect is true",
			args: args{
				cfg:           &kubeadmapi.ResetConfiguration{},
				skipCRIDetect: true,
			},
		},
		{
			name: "CRISocket is empty and skipCRIDetect is false",
			args: args{
				cfg:           &kubeadmapi.ResetConfiguration{},
				skipCRIDetect: false,
			},
		},
		{
			name: "CRISocket is valid",
			args: args{
				cfg: &kubeadmapi.ResetConfiguration{
					CRISocket: "unix:///var/run/containerd/containerd.sock",
				},
				skipCRIDetect: false,
			},
		},
		{
			name: "CRISocket is invalid",
			args: args{
				cfg: &kubeadmapi.ResetConfiguration{
					CRISocket: "var/run/containerd/containerd.sock",
				},
				skipCRIDetect: false,
			},
		},
	}
	for _, rt := range tests {
		t.Run(rt.name, func(t *testing.T) {
			err := SetResetDynamicDefaults(rt.args.cfg, rt.args.skipCRIDetect)
			assert.NoError(t, err)
		})
	}
}
