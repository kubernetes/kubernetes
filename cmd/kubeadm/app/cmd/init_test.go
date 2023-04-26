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
	"fmt"
	"os"
	"path/filepath"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/sets"

	v1 "k8s.io/kubernetes/cmd/kubeadm/app/apis/bootstraptoken/v1"
	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeadmapiv1 "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/v1beta3"
	"k8s.io/kubernetes/cmd/kubeadm/app/cmd/options"
	"k8s.io/kubernetes/cmd/kubeadm/app/constants"
)

var testInitConfig = fmt.Sprintf(`---
apiVersion: %s
kind: InitConfiguration
localAPIEndpoint:
  advertiseAddress: "1.2.3.4"
bootstrapTokens:
- token: "abcdef.0123456789abcdef"
nodeRegistration:
  criSocket: %s
  name: someName
  ignorePreflightErrors:
    - c
    - d
---
apiVersion: %[1]s
kind: ClusterConfiguration
controlPlaneEndpoint: "3.4.5.6"
`, kubeadmapiv1.SchemeGroupVersion.String(), expectedCRISocket)

func TestNewInitData(t *testing.T) {
	// create temp directory
	tmpDir, err := os.MkdirTemp("", "kubeadm-init-test")
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
			name: "pass without any flag except the cri socket (use defaults)",
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
			validate: func(t *testing.T, data *initData) {
				validData := &initData{
					certificatesDir:       kubeadmapiv1.DefaultCertificatesDir,
					kubeconfigPath:        constants.GetAdminKubeConfigPath(),
					kubeconfigDir:         constants.KubernetesDir,
					ignorePreflightErrors: sets.New("c", "d"),
					cfg: &kubeadmapi.InitConfiguration{
						NodeRegistration: kubeadmapi.NodeRegistrationOptions{
							Name:                  "somename",
							CRISocket:             expectedCRISocket,
							IgnorePreflightErrors: []string{"c", "d"},
							ImagePullPolicy:       "IfNotPresent",
						},
						LocalAPIEndpoint: kubeadmapi.APIEndpoint{
							AdvertiseAddress: "1.2.3.4",
							BindPort:         6443,
						},
						BootstrapTokens: []v1.BootstrapToken{
							{
								Token:  &v1.BootstrapTokenString{ID: "abcdef", Secret: "0123456789abcdef"},
								Usages: []string{"signing", "authentication"},
								TTL: &metav1.Duration{
									Duration: constants.DefaultTokenDuration,
								},
								Groups: []string{"system:bootstrappers:kubeadm:default-node-token"},
							},
						},
					},
				}
				if diff := cmp.Diff(validData, data, cmp.AllowUnexported(initData{}), cmpopts.IgnoreFields(initData{}, "client", "cfg.ClusterConfiguration", "cfg.NodeRegistration.Taints")); diff != "" {
					t.Fatalf("newInitData returned data (-want,+got):\n%s", diff)
				}
			},
		},
		{
			name: "--node-name flags override config from file",
			flags: map[string]string{
				options.CfgPath:  configFilePath,
				options.NodeName: "anotherName",
			},
			validate: func(t *testing.T, data *initData) {
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
			cmd := newCmdInit(nil, initOptions)

			// set the cri socket here, otherwise the testcase might fail if is run on the node with multiple
			// cri endpoints configured, the failure caused by this is normally not an expected failure.
			if tc.flags == nil {
				tc.flags = make(map[string]string)
			}
			// set `cri-socket` only if `CfgPath` is not set
			if _, okay := tc.flags[options.CfgPath]; !okay {
				tc.flags[options.NodeCRISocket] = constants.UnknownCRISocket
			}
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
				t.Fatal("newInitData didn't return error when expected")
			}
			// exec additional validation on the returned value
			if tc.validate != nil {
				tc.validate(t, data)
			}
		})
	}
}

func expectedInitIgnorePreflightErrors(expectedItems ...string) func(t *testing.T, data *initData) {
	expected := sets.New(expectedItems...)
	return func(t *testing.T, data *initData) {
		if !expected.Equal(data.ignorePreflightErrors) {
			t.Errorf("Invalid ignore preflight errors. Expected: %v. Actual: %v", sets.List(expected), sets.List(data.ignorePreflightErrors))
		}
		if !expected.HasAll(data.cfg.NodeRegistration.IgnorePreflightErrors...) {
			t.Errorf("Invalid ignore preflight errors in InitConfiguration. Expected: %v. Actual: %v", sets.List(expected), data.cfg.NodeRegistration.IgnorePreflightErrors)
		}
	}
}
