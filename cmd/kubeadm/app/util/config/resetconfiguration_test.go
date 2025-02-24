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
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
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
