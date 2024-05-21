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
	"bytes"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/klog/v2"
	"k8s.io/utils/ptr"

	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeadmapiv1 "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/v1beta3"
	"k8s.io/kubernetes/cmd/kubeadm/app/cmd/options"
	"k8s.io/kubernetes/cmd/kubeadm/app/constants"
	kubeconfigutil "k8s.io/kubernetes/cmd/kubeadm/app/util/kubeconfig"
)

var testJoinConfig = fmt.Sprintf(`apiVersion: %s
kind: JoinConfiguration
discovery:
  bootstrapToken:
    token: abcdef.0123456789abcdef
    apiServerEndpoint: 1.2.3.4:6443
    unsafeSkipCAVerification: true
controlPlane:
  certificateKey: c39a18bae4a72e71b178661f437363da218a3efb83ddb03f1cd91d9ae1da41bd
nodeRegistration:
  criSocket: %s
  name: someName
  ignorePreflightErrors:
    - c
    - d
`, kubeadmapiv1.SchemeGroupVersion.String(), expectedCRISocket)

func TestNewJoinData(t *testing.T) {
	// create temp directory
	tmpDir, err := os.MkdirTemp("", "kubeadm-join-test")
	if err != nil {
		t.Errorf("Unable to create temporary directory: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	// create kubeconfig
	kubeconfigFilePath := filepath.Join(tmpDir, "test-kubeconfig-file")
	kubeconfig := kubeconfigutil.CreateBasic("", "", "", []byte{})
	kubeconfigutil.WriteToDisk(kubeconfigFilePath, kubeconfig)

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
		expectWarn  bool
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
					t.Error("Invalid data.cfg.Discovery.File")
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
					t.Error("Invalid data.cfg.Discovery.BootstrapToken")
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
					t.Error("Invalid TLSBootstrapToken or BootstrapToken.Token")
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
					t.Error("Invalid TLSBootstrapToken or BootstrapToken.Token")
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
					t.Error("Invalid ControlPlane")
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
					t.Error("Invalid ControlPlane")
				}
			},
			expectWarn: true,
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
			validate: func(t *testing.T, data *joinData) {
				validData := &joinData{
					cfg: &kubeadmapi.JoinConfiguration{
						TypeMeta: metav1.TypeMeta{Kind: "", APIVersion: ""},
						NodeRegistration: kubeadmapi.NodeRegistrationOptions{
							Name:                  "somename",
							CRISocket:             expectedCRISocket,
							IgnorePreflightErrors: []string{"c", "d"},
							ImagePullPolicy:       "IfNotPresent",
							ImagePullSerial:       ptr.To(true),
							Taints:                []v1.Taint{{Key: "node-role.kubernetes.io/control-plane", Effect: "NoSchedule"}},
						},
						CACertPath: kubeadmapiv1.DefaultCACertPath,
						Discovery: kubeadmapi.Discovery{
							BootstrapToken: &kubeadmapi.BootstrapTokenDiscovery{
								Token:                    "abcdef.0123456789abcdef",
								APIServerEndpoint:        "1.2.3.4:6443",
								UnsafeSkipCAVerification: true,
							},
							TLSBootstrapToken: "abcdef.0123456789abcdef",
							Timeout:           &metav1.Duration{Duration: constants.DiscoveryTimeout},
						},
						ControlPlane: &kubeadmapi.JoinControlPlane{
							CertificateKey: "c39a18bae4a72e71b178661f437363da218a3efb83ddb03f1cd91d9ae1da41bd",
						},
					},
					ignorePreflightErrors: sets.New("c", "d"),
				}
				if diff := cmp.Diff(validData, data, cmp.AllowUnexported(joinData{}), cmpopts.IgnoreFields(joinData{}, "client", "initCfg", "cfg.ControlPlane.LocalAPIEndpoint", "cfg.Timeouts")); diff != "" {
					t.Fatalf("newJoinData returned data (-want,+got):\n%s", diff)
				}
			},
		},
		{
			name: "--node-name flags override config from file",
			flags: map[string]string{
				options.CfgPath:  configFilePath,
				options.NodeName: "anotherName",
			},
			validate: func(t *testing.T, data *joinData) {
				// validate that node-name is overwritten
				if data.cfg.NodeRegistration.Name != "anotherName" {
					t.Error("Invalid NodeRegistration.Name")
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
			validate: expectedJoinIgnorePreflightErrors(sets.New("a", "b")),
		},
		{
			name: "pre-flights errors from JoinConfiguration only",
			flags: map[string]string{
				options.CfgPath: configFilePath,
			},
			validate: expectedJoinIgnorePreflightErrors(sets.New("c", "d")),
		},
		{
			name: "pre-flights errors from both CLI args and JoinConfiguration",
			flags: map[string]string{
				options.CfgPath:               configFilePath,
				options.IgnorePreflightErrors: "a,b",
			},
			validate: expectedJoinIgnorePreflightErrors(sets.New("a", "b", "c", "d")),
		},
		{
			name: "warn if --control-plane flag is not set",
			flags: map[string]string{
				options.APIServerBindPort: "8888",
				options.FileDiscovery:     "https://foo", //required only to pass discovery validation
			},
			expectWarn: true,
		},
		{
			name: "no warn if --control-plane flag is set",
			flags: map[string]string{
				options.APIServerBindPort: "8888",
				options.FileDiscovery:     "https://bar", //required only to pass discovery validation
				options.ControlPlane:      "true",
			},
		},
	}
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// initialize an external join option and inject it to the join cmd
			joinOptions := newJoinOptions()
			joinOptions.skipCRIDetect = true // avoid CRI detection in unit tests
			cmd := newCmdJoin(nil, joinOptions)

			// set klog output destination to bytes.Buffer so that log could be fetched and verified later.
			var buffer bytes.Buffer
			klog.SetOutput(&buffer)
			klog.LogToStderr(false)
			defer klog.LogToStderr(true)

			// sets cmd flags (that will be reflected on the join options)
			for f, v := range tc.flags {
				cmd.Flags().Set(f, v)
			}

			// test newJoinData method
			data, err := newJoinData(cmd, tc.args, joinOptions, nil, kubeconfigFilePath)
			klog.Flush()
			msg := "WARNING: --control-plane is also required when passing control-plane"
			if tc.expectWarn {
				if !strings.Contains(buffer.String(), msg) {
					t.Errorf("Haven't detected the warning message, expected: %v, actual: %v", msg, buffer.String())
				}
			} else {
				if strings.Contains(buffer.String(), msg) {
					t.Errorf("Expect no such warning message: %v, but got: %v", msg, buffer.String())
				}
			}
			if err != nil && !tc.expectError {
				t.Fatalf("newJoinData returned unexpected error: %v", err)
			}
			if err == nil && tc.expectError {
				t.Fatal("newJoinData didn't return error when expected")
			}

			// exec additional validation on the returned value
			if tc.validate != nil {
				tc.validate(t, data)
			}
		})
	}
}

func expectedJoinIgnorePreflightErrors(expected sets.Set[string]) func(t *testing.T, data *joinData) {
	return func(t *testing.T, data *joinData) {
		if !expected.Equal(data.ignorePreflightErrors) {
			t.Errorf("Invalid ignore preflight errors. Expected: %v. Actual: %v", sets.List(expected), sets.List(data.ignorePreflightErrors))
		}
		if !expected.HasAll(data.cfg.NodeRegistration.IgnorePreflightErrors...) {
			t.Errorf("Invalid ignore preflight errors in JoinConfiguration. Expected: %v. Actual: %v", sets.List(expected), data.cfg.NodeRegistration.IgnorePreflightErrors)
		}
	}
}
