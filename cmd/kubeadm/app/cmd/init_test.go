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

package cmd

import (
	"io/ioutil"
	"os"
	"path/filepath"
	"testing"

	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/kubernetes/cmd/kubeadm/app/cmd/options"
)

const (
	testInitConfig = `---
apiVersion: kubeadm.k8s.io/v1beta2
kind: InitConfiguration
localAPIEndpoint:
  advertiseAddress: "1.2.3.4"
bootstrapTokens:
- token: "abcdef.0123456789abcdef"
nodeRegistration:
  criSocket: /run/containerd/containerd.sock
  name: someName
  ignorePreflightErrors:
    - c
    - d
---
apiVersion: kubeadm.k8s.io/v1beta2
kind: ClusterConfiguration
controlPlaneEndpoint: "3.4.5.6"
`
)

func TestNewInitData(t *testing.T) {
	// create temp directory
	tmpDir, err := ioutil.TempDir("", "kubeadm-init-test")
	if err != nil {
		t.Errorf("Unable to create temporary directory: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	// create config file
	configFilePath := filepath.Join(tmpDir, "test-config-file")
	cfgFile, err := os.Create(configFilePath)
	if err != nil {
		t.Errorf("Unable to create file %q: %v", configFilePath, err)
	}
	defer cfgFile.Close()
	if _, err = cfgFile.WriteString(testInitConfig); err != nil {
		t.Fatalf("Unable to write file %q: %v", configFilePath, err)
	}

	testCases := []struct {
		name        string
		args        []string
		flags       map[string]string
		validate    func(*testing.T, *initData)
		expectError bool
	}{
		// Init data passed using flags
		{
			name: "pass without any flag (use defaults)",
		},
		{
			name: "fail if unknown feature gates flag are passed",
			flags: map[string]string{
				options.FeatureGatesString: "unknown=true",
			},
			expectError: true,
		},
		{
			name: "fails if invalid preflight checks are provided",
			flags: map[string]string{
				options.IgnorePreflightErrors: "all,something-else",
			},
			expectError: true,
		},

		// Init data passed using config file
		{
			name: "Pass with config from file",
			flags: map[string]string{
				options.CfgPath: configFilePath,
			},
		},
		{
			name: "--cri-socket and --node-name flags override config from file",
			flags: map[string]string{
				options.CfgPath:       configFilePath,
				options.NodeCRISocket: "/var/run/crio/crio.sock",
				options.NodeName:      "anotherName",
			},
			validate: func(t *testing.T, data *initData) {
				// validate that cri-socket and node-name are overwritten
				if data.cfg.NodeRegistration.CRISocket != "/var/run/crio/crio.sock" {
					t.Errorf("Invalid NodeRegistration.CRISocket")
				}
				if data.cfg.NodeRegistration.Name != "anotherName" {
					t.Errorf("Invalid NodeRegistration.Name")
				}
			},
		},
		{
			name: "fail if mixedArguments are passed",
			flags: map[string]string{
				options.CfgPath:                   configFilePath,
				options.APIServerAdvertiseAddress: "1.2.3.4",
			},
			expectError: true,
		},

		// Pre-flight errors:
		{
			name: "pre-flights errors from CLI args only",
			flags: map[string]string{
				options.IgnorePreflightErrors: "a,b",
			},
			validate: expectedInitIgnorePreflightErrors("a", "b"),
		},
		{
			name: "pre-flights errors from InitConfiguration only",
			flags: map[string]string{
				options.CfgPath: configFilePath,
			},
			validate: expectedInitIgnorePreflightErrors("c", "d"),
		},
		{
			name: "pre-flights errors from both CLI args and InitConfiguration",
			flags: map[string]string{
				options.CfgPath:               configFilePath,
				options.IgnorePreflightErrors: "a,b",
			},
			validate: expectedInitIgnorePreflightErrors("a", "b", "c", "d"),
		},
	}
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// initialize an external init option and inject it to the init cmd
			initOptions := newInitOptions()
			cmd := NewCmdInit(nil, initOptions)

			// sets cmd flags (that will be reflected on the init options)
			for f, v := range tc.flags {
				cmd.Flags().Set(f, v)
			}

			// test newInitData method
			data, err := newInitData(cmd, tc.args, initOptions, nil)
			if err != nil && !tc.expectError {
				t.Fatalf("newInitData returned unexpected error: %v", err)
			}
			if err == nil && tc.expectError {
				t.Fatalf("newInitData didn't return error when expected")
			}

			// exec additional validation on the returned value
			if tc.validate != nil {
				tc.validate(t, data)
			}
		})
	}
}

func expectedInitIgnorePreflightErrors(expectedItems ...string) func(t *testing.T, data *initData) {
	expected := sets.NewString(expectedItems...)
	return func(t *testing.T, data *initData) {
		if !expected.Equal(data.ignorePreflightErrors) {
			t.Errorf("Invalid ignore preflight errors. Expected: %v. Actual: %v", expected.List(), data.ignorePreflightErrors.List())
		}
		if !expected.HasAll(data.cfg.NodeRegistration.IgnorePreflightErrors...) {
			t.Errorf("Invalid ignore preflight errors in InitConfiguration. Expected: %v. Actual: %v", expected.List(), data.cfg.NodeRegistration.IgnorePreflightErrors)
		}
	}
}
