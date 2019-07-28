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
	testJoinConfig = `apiVersion: kubeadm.k8s.io/v1beta2
kind: JoinConfiguration
discovery:
  bootstrapToken:
    token: abcdef.0123456789abcdef
    apiServerEndpoint: 1.2.3.4:6443
    unsafeSkipCAVerification: true
nodeRegistration:
  criSocket: /run/containerd/containerd.sock
  name: someName
  ignorePreflightErrors:
    - c
    - d
`
)

func TestNewJoinData(t *testing.T) {
	// create temp directory
	tmpDir, err := ioutil.TempDir("", "kubeadm-join-test")
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
	if _, err = cfgFile.WriteString(testJoinConfig); err != nil {
		t.Fatalf("Unable to write file %q: %v", configFilePath, err)
	}

	testCases := []struct {
		name        string
		args        []string
		flags       map[string]string
		validate    func(*testing.T, *joinData)
		expectError bool
	}{
		// Join data passed using flags
		{
			name:        "fails if no discovery method set",
			expectError: true,
		},
		{
			name: "fails if both file and bootstrap discovery methods set",
			args: []string{"1.2.3.4:6443"},
			flags: map[string]string{
				options.FileDiscovery:            "https://foo",
				options.TokenDiscovery:           "abcdef.0123456789abcdef",
				options.TokenDiscoverySkipCAHash: "true",
			},
			expectError: true,
		},
		{
			name: "pass if file discovery is set",
			flags: map[string]string{
				options.FileDiscovery: "https://foo",
			},
			validate: func(t *testing.T, data *joinData) {
				// validate that file discovery settings are set into join data
				if data.cfg.Discovery.File == nil || data.cfg.Discovery.File.KubeConfigPath != "https://foo" {
					t.Errorf("Invalid data.cfg.Discovery.File")
				}
			},
		},
		{
			name: "pass if bootstrap discovery is set",
			args: []string{"1.2.3.4:6443", "5.6.7.8:6443"},
			flags: map[string]string{
				options.TokenDiscovery:           "abcdef.0123456789abcdef",
				options.TokenDiscoverySkipCAHash: "true",
			},
			validate: func(t *testing.T, data *joinData) {
				// validate that bootstrap discovery settings are set into join data
				if data.cfg.Discovery.BootstrapToken == nil ||
					data.cfg.Discovery.BootstrapToken.APIServerEndpoint != "1.2.3.4:6443" || //only first arg should be kept as APIServerEndpoint
					data.cfg.Discovery.BootstrapToken.Token != "abcdef.0123456789abcdef" ||
					data.cfg.Discovery.BootstrapToken.UnsafeSkipCAVerification != true {
					t.Errorf("Invalid data.cfg.Discovery.BootstrapToken")
				}
			},
		},
		{
			name: "--token sets TLSBootstrapToken and BootstrapToken.Token if unset",
			args: []string{"1.2.3.4:6443"},
			flags: map[string]string{
				options.TokenStr:                 "abcdef.0123456789abcdef",
				options.TokenDiscoverySkipCAHash: "true",
			},
			validate: func(t *testing.T, data *joinData) {
				// validate that token sets both TLSBootstrapToken and BootstrapToken.Token into join data
				if data.cfg.Discovery.TLSBootstrapToken != "abcdef.0123456789abcdef" ||
					data.cfg.Discovery.BootstrapToken == nil ||
					data.cfg.Discovery.BootstrapToken.Token != "abcdef.0123456789abcdef" {
					t.Errorf("Invalid TLSBootstrapToken or BootstrapToken.Token")
				}
			},
		},
		{
			name: "--token doesn't override TLSBootstrapToken and BootstrapToken.Token if set",
			args: []string{"1.2.3.4:6443"},
			flags: map[string]string{
				options.TokenStr:                 "aaaaaa.0123456789aaaaaa",
				options.TLSBootstrapToken:        "abcdef.0123456789abcdef",
				options.TokenDiscovery:           "defghi.0123456789defghi",
				options.TokenDiscoverySkipCAHash: "true",
			},
			validate: func(t *testing.T, data *joinData) {
				// validate that TLSBootstrapToken and BootstrapToken.Token values are preserved into join data
				if data.cfg.Discovery.TLSBootstrapToken != "abcdef.0123456789abcdef" ||
					data.cfg.Discovery.BootstrapToken == nil ||
					data.cfg.Discovery.BootstrapToken.Token != "defghi.0123456789defghi" {
					t.Errorf("Invalid TLSBootstrapToken or BootstrapToken.Token")
				}
			},
		},
		{
			name: "control plane setting are preserved if --control-plane flag is set",
			flags: map[string]string{
				options.ControlPlane:              "true",
				options.APIServerAdvertiseAddress: "1.2.3.4",
				options.APIServerBindPort:         "1234",
				options.FileDiscovery:             "https://foo", //required only to pass discovery validation
			},
			validate: func(t *testing.T, data *joinData) {
				// validate that control plane attributes are set in join data
				if data.cfg.ControlPlane == nil ||
					data.cfg.ControlPlane.LocalAPIEndpoint.AdvertiseAddress != "1.2.3.4" ||
					data.cfg.ControlPlane.LocalAPIEndpoint.BindPort != 1234 {
					t.Errorf("Invalid ControlPlane")
				}
			},
		},
		{
			name: "control plane setting are cleaned up if --control-plane flag is not set",
			flags: map[string]string{
				options.ControlPlane:              "false",
				options.APIServerAdvertiseAddress: "1.2.3.4",
				options.APIServerBindPort:         "1.2.3.4",
				options.FileDiscovery:             "https://foo", //required only to pass discovery validation
			},
			validate: func(t *testing.T, data *joinData) {
				// validate that control plane attributes are unset in join data
				if data.cfg.ControlPlane != nil {
					t.Errorf("Invalid ControlPlane")
				}
			},
		},
		{
			name: "fails if invalid preflight checks are provided",
			flags: map[string]string{
				options.IgnorePreflightErrors: "all,something-else",
			},
			expectError: true,
		},

		// Join data passed using config file
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
			validate: func(t *testing.T, data *joinData) {
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
				options.FileDiscovery:         "https://foo", //required only to pass discovery validation
			},
			validate: expectedJoinIgnorePreflightErrors(sets.NewString("a", "b")),
		},
		{
			name: "pre-flights errors from JoinConfiguration only",
			flags: map[string]string{
				options.CfgPath: configFilePath,
			},
			validate: expectedJoinIgnorePreflightErrors(sets.NewString("c", "d")),
		},
		{
			name: "pre-flights errors from both CLI args and JoinConfiguration",
			flags: map[string]string{
				options.CfgPath:               configFilePath,
				options.IgnorePreflightErrors: "a,b",
			},
			validate: expectedJoinIgnorePreflightErrors(sets.NewString("a", "b", "c", "d")),
		},
	}
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// initialize an external join option and inject it to the join cmd
			joinOptions := newJoinOptions()
			cmd := NewCmdJoin(nil, joinOptions)

			// sets cmd flags (that will be reflected on the join options)
			for f, v := range tc.flags {
				cmd.Flags().Set(f, v)
			}

			// test newJoinData method
			data, err := newJoinData(cmd, tc.args, joinOptions, nil)
			if err != nil && !tc.expectError {
				t.Fatalf("newJoinData returned unexpected error: %v", err)
			}
			if err == nil && tc.expectError {
				t.Fatalf("newJoinData didn't return error when expected")
			}

			// exec additional validation on the returned value
			if tc.validate != nil {
				tc.validate(t, data)
			}
		})
	}
}

func expectedJoinIgnorePreflightErrors(expected sets.String) func(t *testing.T, data *joinData) {
	return func(t *testing.T, data *joinData) {
		if !expected.Equal(data.ignorePreflightErrors) {
			t.Errorf("Invalid ignore preflight errors. Expected: %v. Actual: %v", expected.List(), data.ignorePreflightErrors.List())
		}
		if !expected.HasAll(data.cfg.NodeRegistration.IgnorePreflightErrors...) {
			t.Errorf("Invalid ignore preflight errors in JoinConfiguration. Expected: %v. Actual: %v", expected.List(), data.cfg.NodeRegistration.IgnorePreflightErrors)
		}
	}
}
