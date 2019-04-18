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

	"github.com/spf13/cobra"
	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeadmapiv1beta2 "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/v1beta2"
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

func initJoinOptions(token string, discovery kubeadmapiv1beta2.Discovery, controlPlane bool) *joinOptions {
	joinOptions := newJoinOptions()
	joinOptions.token = token
	joinOptions.externalcfg.Discovery = discovery
	joinOptions.controlPlane = controlPlane
	return joinOptions
}

func TestAmendExternalJoinConfiguration(t *testing.T) {
	var tests = []struct {
		description  string
		args         []string
		flags        map[string]string
		token        string
		controlPlane bool
		discovery    kubeadmapiv1beta2.Discovery
		expectedErr  bool
		validate     func(*testing.T, *joinOptions)
	}{
		{
			description: "len(opt.token) = 0",
			args:        []string{"1.2.3.4:6443"},
			flags:       map[string]string{options.FileDiscovery: "https://foo"},
			token:       "",
			discovery: kubeadmapiv1beta2.Discovery{
				BootstrapToken: &kubeadmapiv1beta2.BootstrapTokenDiscovery{
					Token: "abcdef.0123456789abcdef",
				},
				File:              &kubeadmapiv1beta2.FileDiscovery{},
				TLSBootstrapToken: "abcdef.0123456789",
			},
			expectedErr: false,
			validate: func(t *testing.T, opt *joinOptions) {
				if opt.externalcfg.Discovery.TLSBootstrapToken != "abcdef.0123456789" {
					t.Errorf("opt.externalcfg.Discovery.TLSBootstrapToken shouldn't be set to %s", opt.externalcfg.Discovery.TLSBootstrapToken)
				}
				if opt.externalcfg.Discovery.BootstrapToken.Token != "abcdef.0123456789abcdef" {
					t.Errorf("opt.externalcfg.Discovery.BootstrapToken.Token shouldn't be set to %s", opt.externalcfg.Discovery.BootstrapToken.Token)
				}
			},
		},
		{
			description: "len(opt.token) > 0",
			args:        []string{"1.2.3.4:6443"},
			flags:       map[string]string{options.FileDiscovery: "https://foo"},
			token:       "abcdef.0123456789abcdef",
			discovery: kubeadmapiv1beta2.Discovery{
				BootstrapToken:    &kubeadmapiv1beta2.BootstrapTokenDiscovery{},
				File:              &kubeadmapiv1beta2.FileDiscovery{},
				TLSBootstrapToken: "",
			},
			expectedErr: false,
			validate: func(t *testing.T, opt *joinOptions) {
				if opt.externalcfg.Discovery.TLSBootstrapToken != opt.token {
					t.Errorf("opt.externalcfg.Discovery.TLSBootstrapToken should be set to %s", opt.token)
				}
				if opt.externalcfg.Discovery.BootstrapToken.Token != opt.token {
					t.Errorf("opt.externalcfg.Discovery.BootstrapToken.Token should be set to %s", opt.token)
				}
			},
		},
		{
			description: "len(args) == 0",
			args:        []string{},
			flags:       map[string]string{options.FileDiscovery: "https://foo"},
			token:       "abcdef.0123456789abcdef",
			discovery: kubeadmapiv1beta2.Discovery{
				BootstrapToken: &kubeadmapiv1beta2.BootstrapTokenDiscovery{},
				File:           &kubeadmapiv1beta2.FileDiscovery{},
			},
			expectedErr: false,
			validate: func(t *testing.T, opt *joinOptions) {
				if opt.externalcfg.Discovery.BootstrapToken != nil {
					t.Errorf("opt.externalcfg.Discovery.BootstrapToken should be set nil when len(args) == 0")
				}
			},
		},
		{
			description: "len(args) > 0",
			args:        []string{"1.2.3.4:6443"},
			flags:       map[string]string{options.FileDiscovery: "https://foo"},
			token:       "abcdef.0123456789abcdef",
			discovery: kubeadmapiv1beta2.Discovery{
				BootstrapToken: &kubeadmapiv1beta2.BootstrapTokenDiscovery{},
				File:           &kubeadmapiv1beta2.FileDiscovery{},
			},
			expectedErr: false,
			validate: func(t *testing.T, opt *joinOptions) {
				if opt.externalcfg.Discovery.BootstrapToken.APIServerEndpoint != "1.2.3.4:6443" {
					t.Errorf("opt.externalcfg.Discovery.BootstrapToken.APIServerEndpoint should be set args[0]")
				}
			},
		},
		{
			description: "len(args) > 1",
			args:        []string{"1.2.3.4:6443", "5.6.7.8:6443"},
			flags:       map[string]string{options.FileDiscovery: "https://foo"},
			token:       "abcdef.0123456789abcdef",
			discovery: kubeadmapiv1beta2.Discovery{
				BootstrapToken: &kubeadmapiv1beta2.BootstrapTokenDiscovery{},
				File:           &kubeadmapiv1beta2.FileDiscovery{},
			},
			expectedErr: false,
			validate: func(t *testing.T, opt *joinOptions) {
				if opt.externalcfg.Discovery.BootstrapToken.APIServerEndpoint != "1.2.3.4:6443" {
					t.Errorf("opt.externalcfg.Discovery.BootstrapToken.APIServerEndpoint should be set args[0]")
				}
			},
		},
		{
			description: "len(opt.externalcfg.Discovery.File.KubeConfigPath) == 0",
			args:        []string{"1.2.3.4:6443"},
			flags:       map[string]string{options.FileDiscovery: "https://foo"},
			token:       "abcdef.0123456789abcdef",
			discovery: kubeadmapiv1beta2.Discovery{
				BootstrapToken: &kubeadmapiv1beta2.BootstrapTokenDiscovery{},
				File:           &kubeadmapiv1beta2.FileDiscovery{KubeConfigPath: ""},
			},
			expectedErr: false,
			validate: func(t *testing.T, opt *joinOptions) {
				if opt.externalcfg.Discovery.File != nil {
					t.Errorf("opt.externalcfg.Discovery.File should be set nil when len(File.KubeConfigPath) == 0")
				}
			},
		},
		{
			description: "opt.controlPlane is false",
			args:        []string{"1.2.3.4:6443"},
			flags:       map[string]string{options.FileDiscovery: "https://foo"},
			token:       "abcdef.0123456789abcdef",
			discovery: kubeadmapiv1beta2.Discovery{
				BootstrapToken: &kubeadmapiv1beta2.BootstrapTokenDiscovery{},
				File:           &kubeadmapiv1beta2.FileDiscovery{},
			},
			controlPlane: false,
			expectedErr:  false,
			validate: func(t *testing.T, opt *joinOptions) {
				if opt.externalcfg.ControlPlane != nil {
					t.Errorf("opt.externalcfg.ControlPlane should be set nil when opt.controlPlane is false")
				}
			},
		},
		{
			description: "cmd.Flags().Lookup(options.FileDiscovery) == nil",
			args:        []string{"1.2.3.4:6443"},
			flags:       map[string]string{},
			token:       "abcdef.0123456789abcdef",
			discovery: kubeadmapiv1beta2.Discovery{
				BootstrapToken: &kubeadmapiv1beta2.BootstrapTokenDiscovery{},
				File:           &kubeadmapiv1beta2.FileDiscovery{},
			},
			expectedErr: false,
			validate: func(t *testing.T, opt *joinOptions) {
				if opt.externalcfg.Discovery.BootstrapToken != nil {
					t.Errorf("opt.externalcfg.Discovery.BootstrapToken should be set nil")
				}
			},
		},
	}
	for _, rt := range tests {
		t.Run(rt.description, func(t *testing.T) {
			joinOptions := initJoinOptions(rt.token, rt.discovery, rt.controlPlane)

			cmd := &cobra.Command{}
			for f, v := range rt.flags {
				cmd.Flags().String(f, v, "")
			}

			adminKubeConfigPath, err := ioutil.TempDir("", "kubeadm-jointest")
			if err != nil {
				t.Errorf("couldn't create a temporary directory: %v", err)
			}

			actualErr := amendExternalJoinConfiguration(joinOptions, rt.args, cmd, adminKubeConfigPath)
			if (actualErr != nil) != rt.expectedErr {
				t.Errorf("%s failed, expectedErr: %v\n\t actualErr: %v", rt.description, rt.expectedErr, actualErr)
			}

			if rt.validate != nil {
				rt.validate(t, joinOptions)
			}
		})
	}
}

func TestLoadInternalJoinConfiguration(t *testing.T) {
	var tests = []struct {
		description string
		opt         *joinOptions
		validate    func(*testing.T, *kubeadmapi.JoinConfiguration)
		expectedErr bool
	}{
		{
			description: "configutil.LoadOrDefaultJoinConfiguration failed should return error",
			opt: &joinOptions{
				cfgPath: "/tmp/cfgPath",
			},
			expectedErr: true,
		},
		{
			description: "set cfg.NodeRegistration name and CRISocket",
			opt: &joinOptions{
				externalcfg: &kubeadmapiv1beta2.JoinConfiguration{
					NodeRegistration: kubeadmapiv1beta2.NodeRegistrationOptions{
						Name:      "anotherName",
						CRISocket: "/var/run/crio/crio.sock",
					},
					Discovery: kubeadmapiv1beta2.Discovery{
						BootstrapToken: &kubeadmapiv1beta2.BootstrapTokenDiscovery{
							Token:                    "abcdef.0123456789abcdef",
							APIServerEndpoint:        "1.2.3.4:6443",
							UnsafeSkipCAVerification: true,
						},
					},
				},
			},
			validate: func(t *testing.T, cfg *kubeadmapi.JoinConfiguration) {
				if cfg.NodeRegistration.Name != "anotherName" {
					t.Errorf("cfg.NodeRegistration.Name should be set 'anotherName'")
				}
				if cfg.NodeRegistration.CRISocket != "/var/run/crio/crio.sock" {
					t.Errorf("cfg.NodeRegistration.CRISocket should be set '/var/run/crio/crio.sock'")
				}
			},
			expectedErr: false,
		},
	}
	for _, rt := range tests {
		t.Run(rt.description, func(t *testing.T) {
			cfg, actualErr := loadInternalJoinConfiguration(rt.opt)
			if (actualErr != nil) != rt.expectedErr {
				t.Errorf("%s failed, expectedErr: %v\n\t actualErr: %v", rt.description, rt.expectedErr, actualErr)
			}
			if cfg != nil && rt.validate != nil {
				rt.validate(t, cfg)
			}
		})
	}
}
