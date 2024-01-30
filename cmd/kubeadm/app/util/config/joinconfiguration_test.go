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
	"os"
	"path/filepath"
	"testing"

	"github.com/lithammer/dedent"
)

func TestLoadJoinConfigurationFromFile(t *testing.T) {
	// Create temp folder for the test case
	tmpdir, err := os.MkdirTemp("", "")
	if err != nil {
		t.Fatalf("Couldn't create tmpdir: %v", err)
	}
	defer os.RemoveAll(tmpdir)

	// cfgFiles is in cluster_test.go
	var tests = []struct {
		name         string
		fileContents string
		expectErr    bool
	}{
		{
			name:      "empty file causes error",
			expectErr: true,
		},
		{
			name: "Invalid v1beta3 causes error",
			fileContents: dedent.Dedent(`
				apiVersion: kubeadm.k8s.io/v1beta3
				kind: JoinConfiguration
			`),
			expectErr: true,
		},
		{
			name: "valid v1beta3 is loaded",
			fileContents: dedent.Dedent(`
				apiVersion: kubeadm.k8s.io/v1beta3
				kind: JoinConfiguration
				caCertPath: /etc/kubernetes/pki/ca.crt
				nodeRegistration:
				  criSocket: "unix:///var/run/unknown.sock"
				discovery:
				  bootstrapToken:
				    apiServerEndpoint: kube-apiserver:6443
				    token: abcdef.0123456789abcdef
				    unsafeSkipCAVerification: true
				  timeout: 5m0s
				  tlsBootstrapToken: abcdef.0123456789abcdef
			`),
		},
	}

	for _, rt := range tests {
		t.Run(rt.name, func(t2 *testing.T) {
			cfgPath := filepath.Join(tmpdir, rt.name)
			err := os.WriteFile(cfgPath, []byte(rt.fileContents), 0644)
			if err != nil {
				t.Errorf("Couldn't create file: %v", err)
				return
			}

			opts := LoadOrDefaultConfigurationOptions{
				SkipCRIDetect: true,
			}

			obj, err := LoadJoinConfigurationFromFile(cfgPath, opts)
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
		})
	}
}
