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

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/utils/ptr"

	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeadmapiv1 "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/v1beta4"
	"k8s.io/kubernetes/cmd/kubeadm/app/constants"
)

func TestBytesToJoinConfiguration(t *testing.T) {
	options := LoadOrDefaultConfigurationOptions{}

	tests := []struct {
		name string
		cfg  *kubeadmapiv1.JoinConfiguration
		want *kubeadmapi.JoinConfiguration
	}{
		{
			name: "Normal configuration",
			cfg: &kubeadmapiv1.JoinConfiguration{
				TypeMeta: metav1.TypeMeta{
					APIVersion: kubeadmapiv1.SchemeGroupVersion.String(),
					Kind:       constants.JoinConfigurationKind,
				},
				NodeRegistration: kubeadmapiv1.NodeRegistrationOptions{
					Name:      "node-1",
					CRISocket: "unix:///var/run/crio/crio.sock",
				},
				CACertPath: "/some/cert.crt",
				Discovery: kubeadmapiv1.Discovery{
					BootstrapToken: &kubeadmapiv1.BootstrapTokenDiscovery{
						Token:             "abcdef.1234567890123456",
						APIServerEndpoint: "1.2.3.4:6443",
						CACertHashes:      []string{"aaaa"},
					},
					TLSBootstrapToken: "abcdef.1234567890123456",
				},
			},
			want: &kubeadmapi.JoinConfiguration{
				NodeRegistration: kubeadmapi.NodeRegistrationOptions{
					Name:            "node-1",
					CRISocket:       "unix:///var/run/crio/crio.sock",
					ImagePullPolicy: "IfNotPresent",
					ImagePullSerial: ptr.To(true),
				},
				CACertPath: "/some/cert.crt",
				Discovery: kubeadmapi.Discovery{
					BootstrapToken: &kubeadmapi.BootstrapTokenDiscovery{
						Token:             "abcdef.1234567890123456",
						APIServerEndpoint: "1.2.3.4:6443",
						CACertHashes:      []string{"aaaa"},
					},
					TLSBootstrapToken: "abcdef.1234567890123456",
				},
				ControlPlane: nil,
			},
		},
		{
			name: "Only contains Discovery configuration",
			cfg: &kubeadmapiv1.JoinConfiguration{
				TypeMeta: metav1.TypeMeta{
					APIVersion: kubeadmapiv1.SchemeGroupVersion.String(),
					Kind:       constants.JoinConfigurationKind,
				},
				Discovery: kubeadmapiv1.Discovery{
					BootstrapToken: &kubeadmapiv1.BootstrapTokenDiscovery{
						Token:             "abcdef.1234567890123456",
						APIServerEndpoint: "1.2.3.4:6443",
						CACertHashes:      []string{"aaaa"},
					},
					TLSBootstrapToken: "abcdef.1234567890123456",
				},
			},
			want: &kubeadmapi.JoinConfiguration{
				NodeRegistration: kubeadmapi.NodeRegistrationOptions{
					CRISocket:       "unix:///var/run/containerd/containerd.sock",
					ImagePullPolicy: "IfNotPresent",
					ImagePullSerial: ptr.To(true),
				},
				CACertPath: "/etc/kubernetes/pki/ca.crt",
				Discovery: kubeadmapi.Discovery{
					BootstrapToken: &kubeadmapi.BootstrapTokenDiscovery{
						Token:             "abcdef.1234567890123456",
						APIServerEndpoint: "1.2.3.4:6443",
						CACertHashes:      []string{"aaaa"},
					},
					TLSBootstrapToken: "abcdef.1234567890123456",
				},
				ControlPlane: nil,
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

				got, _ := BytesToJoinConfiguration(bytes, options)
				if diff := cmp.Diff(got, tt.want, cmpopts.IgnoreFields(kubeadmapi.JoinConfiguration{}, "Timeouts"),
					cmpopts.IgnoreFields(kubeadmapi.Discovery{}, "Timeout"), cmpopts.IgnoreFields(kubeadmapi.NodeRegistrationOptions{}, "Name")); diff != "" {
					t.Errorf("LoadJoinConfigurationFromFile returned unexpected diff (-want,+got):\n%s", diff)
				}
			})
		}
	}
}

func TestLoadJoinConfigurationFromFile(t *testing.T) {
	// Create temp folder for the test case
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
discovery:
  bootstrapToken:
    apiServerEndpoint: 1.2.3.4:6443
    caCertHashes:
    - aaaa
    token: abcdef.1234567890123456
  tlsBootstrapToken: abcdef.1234567890123456
kind: JoinConfiguration`),
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

			_, err = LoadJoinConfigurationFromFile(tt.cfgPath, options)
			if (err != nil) != tt.wantErr {
				t.Errorf("LoadJoinConfigurationFromFile() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}
