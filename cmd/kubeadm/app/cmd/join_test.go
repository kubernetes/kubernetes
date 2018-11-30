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
	"io/ioutil"
	"os"
	"path/filepath"
	"testing"

	"k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeadmscheme "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/scheme"
	kubeadmapiv1beta1 "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/v1beta1"
)

const (
	testConfig = `apiVersion: v1
clusters:
- cluster:
    certificate-authority-data:
    server: localhost:9008
  name: prod
contexts:
- context:
    cluster: prod
    namespace: default
    user: default-service-account
  name: default
current-context: default
kind: Config
preferences: {}
users:
- name: kubernetes-admin
  user:
    client-certificate-data:
    client-key-data:
`
)

func TestNewValidJoin(t *testing.T) {
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

	testCases := []struct {
		name                  string
		skipPreFlight         bool
		cfgPath               string
		configToWrite         string
		ignorePreflightErrors []string
		testJoinValidate      bool
		testJoinRun           bool
		cmdPersistentFlags    map[string]string
		nodeConfig            *kubeadm.JoinConfiguration
		expectedError         bool
	}{
		{
			name:          "invalid: missing config file",
			skipPreFlight: true,
			cfgPath:       "missing-path-to-a-config",
			expectedError: true,
		},
		{
			name:          "invalid: incorrect config file",
			skipPreFlight: true,
			cfgPath:       configFilePath,
			configToWrite: "bad-config-contents",
			expectedError: true,
		},
		{
			name:          "invalid: fail at preflight.RunJoinNodeChecks()",
			skipPreFlight: false,
			cfgPath:       configFilePath,
			configToWrite: testConfig,
			expectedError: true,
		},
		{
			name:                  "invalid: incorrect ignorePreflight argument",
			skipPreFlight:         true,
			cfgPath:               configFilePath,
			configToWrite:         testConfig,
			ignorePreflightErrors: []string{"some-unsupported-preflight-arg"},
			expectedError:         true,
		},
		{
			name:             "invalid: fail Join.Validate() with wrong flags",
			skipPreFlight:    true,
			cfgPath:          configFilePath,
			configToWrite:    testConfig,
			testJoinValidate: true,
			cmdPersistentFlags: map[string]string{
				"config":    "some-config",
				"node-name": "some-node-name",
			},
			expectedError: true,
		},
		{
			name:             "invalid: fail Join.Validate() with wrong node configuration",
			skipPreFlight:    true,
			cfgPath:          configFilePath,
			configToWrite:    testConfig,
			testJoinValidate: true,
			expectedError:    true,
		},
		{
			name:          "invalid: fail Join.Run() with invalid node config",
			skipPreFlight: true,
			cfgPath:       configFilePath,
			configToWrite: testConfig,
			testJoinRun:   true,
			expectedError: true,
		},
	}

	var out bytes.Buffer
	cfg := &kubeadmapiv1beta1.JoinConfiguration{}
	kubeadmscheme.Scheme.Default(cfg)

	errorFormat := "Test case %q: NewValidJoin expected error: %v, saw: %v, error: %v"

	for _, tc := range testCases {
		if _, err = cfgFile.WriteString(tc.configToWrite); err != nil {
			t.Fatalf("Unable to write file %q: %v", tc.cfgPath, err)
		}

		cmd := NewCmdJoin(&out)
		if tc.cmdPersistentFlags != nil {
			for key, value := range tc.cmdPersistentFlags {
				cmd.PersistentFlags().Set(key, value)
			}
		}

		join, err := NewValidJoin(cmd.PersistentFlags(), cfg, tc.cfgPath, tc.ignorePreflightErrors)

		if tc.nodeConfig != nil {
			join.cfg = tc.nodeConfig
		}

		// test Join.Run()
		if err == nil && tc.testJoinRun {
			err = join.Run(&out)
			if (err != nil) != tc.expectedError {
				t.Fatalf(errorFormat, tc.name, tc.expectedError, (err != nil), err)
			}
			// check error for NewValidJoin()
		} else if (err != nil) != tc.expectedError {
			t.Fatalf(errorFormat, tc.name, tc.expectedError, (err != nil), err)
		}
	}
}
