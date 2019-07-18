/*
Copyright 2016 The Kubernetes Authors.

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
	"regexp"
	"testing"

	"k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/kubernetes/fake"
	core "k8s.io/client-go/testing"
	"k8s.io/client-go/tools/clientcmd"
	kubeadmapiv1beta2 "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/v1beta2"
)

const (
	tokenExpectedRegex = "^\\S{6}\\.\\S{16}\n$"
	testConfigToken    = `apiVersion: v1
clusters:
- cluster:
    certificate-authority-data:
    server: localhost:8000
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

func TestRunGenerateToken(t *testing.T) {
	var buf bytes.Buffer

	err := RunGenerateToken(&buf)
	if err != nil {
		t.Errorf("RunGenerateToken returned an error: %v", err)
	}

	output := buf.String()

	matched, err := regexp.MatchString(tokenExpectedRegex, output)
	if err != nil {
		t.Fatalf("Encountered an error while trying to match RunGenerateToken's output: %v", err)
	}
	if !matched {
		t.Errorf("RunGenerateToken's output did not match expected regex; wanted: [%s], got: [%s]", tokenExpectedRegex, output)
	}
}

func TestRunCreateToken(t *testing.T) {
	var buf bytes.Buffer
	fakeClient := &fake.Clientset{}
	fakeClient.AddReactor("get", "secrets", func(action core.Action) (handled bool, ret runtime.Object, err error) {
		return true, nil, apierrors.NewNotFound(v1.Resource("secrets"), "foo")
	})

	testCases := []struct {
		name          string
		token         string
		usages        []string
		extraGroups   []string
		printJoin     bool
		expectedError bool
	}{
		{
			name:          "valid: empty token",
			token:         "",
			usages:        []string{"signing", "authentication"},
			extraGroups:   []string{"system:bootstrappers:foo"},
			expectedError: false,
		},
		{
			name:          "valid: non-empty token",
			token:         "abcdef.1234567890123456",
			usages:        []string{"signing", "authentication"},
			extraGroups:   []string{"system:bootstrappers:foo"},
			expectedError: false,
		},
		{
			name:          "valid: no extraGroups",
			token:         "abcdef.1234567890123456",
			usages:        []string{"signing", "authentication"},
			extraGroups:   []string{},
			expectedError: false,
		},
		{
			name:          "invalid: incorrect extraGroups",
			token:         "abcdef.1234567890123456",
			usages:        []string{"signing", "authentication"},
			extraGroups:   []string{"foo"},
			expectedError: true,
		},
		{
			name:          "invalid: specifying --groups when --usages doesn't include authentication",
			token:         "abcdef.1234567890123456",
			usages:        []string{"signing"},
			extraGroups:   []string{"foo"},
			expectedError: true,
		},
		{
			name:          "invalid: partially incorrect usages",
			token:         "abcdef.1234567890123456",
			usages:        []string{"foo", "authentication"},
			extraGroups:   []string{"system:bootstrappers:foo"},
			expectedError: true,
		},
		{
			name:          "invalid: all incorrect usages",
			token:         "abcdef.1234567890123456",
			usages:        []string{"foo", "bar"},
			extraGroups:   []string{"system:bootstrappers:foo"},
			expectedError: true,
		},
		{
			name:          "invalid: print join command",
			token:         "",
			usages:        []string{"signing", "authentication"},
			extraGroups:   []string{"system:bootstrappers:foo"},
			printJoin:     true,
			expectedError: true,
		},
	}
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			bts, err := kubeadmapiv1beta2.NewBootstrapTokenString(tc.token)
			if err != nil && len(tc.token) != 0 { // if tc.token is "" it's okay as it will be generated later at runtime
				t.Fatalf("token couldn't be parsed for testing: %v", err)
			}

			cfg := &kubeadmapiv1beta2.InitConfiguration{
				BootstrapTokens: []kubeadmapiv1beta2.BootstrapToken{
					{
						Token:  bts,
						TTL:    &metav1.Duration{Duration: 0},
						Usages: tc.usages,
						Groups: tc.extraGroups,
					},
				},
			}

			err = RunCreateToken(&buf, fakeClient, "", cfg, tc.printJoin, "")
			if tc.expectedError && err == nil {
				t.Error("unexpected success")
			} else if !tc.expectedError && err != nil {
				t.Errorf("unexpected error: %v", err)
			}
		})
	}
}

func TestNewCmdTokenGenerate(t *testing.T) {
	var buf bytes.Buffer
	args := []string{}

	cmd := NewCmdTokenGenerate(&buf)
	cmd.SetArgs(args)

	if err := cmd.Execute(); err != nil {
		t.Errorf("Cannot execute token command: %v", err)
	}
}

func TestNewCmdToken(t *testing.T) {
	var buf, bufErr bytes.Buffer
	testConfigTokenFile := "test-config-file"

	tmpDir, err := ioutil.TempDir("", "kubeadm-token-test")
	if err != nil {
		t.Errorf("Unable to create temporary directory: %v", err)
	}
	defer os.RemoveAll(tmpDir)
	fullPath := filepath.Join(tmpDir, testConfigTokenFile)

	f, err := os.Create(fullPath)
	if err != nil {
		t.Errorf("Unable to create test file %q: %v", fullPath, err)
	}
	defer f.Close()

	testCases := []struct {
		name          string
		args          []string
		configToWrite string
		kubeConfigEnv string
		expectedError bool
	}{
		{
			name:          "valid: generate",
			args:          []string{"generate"},
			configToWrite: "",
			expectedError: false,
		},
		{
			name:          "valid: delete from --kubeconfig",
			args:          []string{"delete", "abcdef.1234567890123456", "--dry-run", "--kubeconfig=" + fullPath},
			configToWrite: testConfigToken,
			expectedError: false,
		},
		{
			name:          "valid: delete from " + clientcmd.RecommendedConfigPathEnvVar,
			args:          []string{"delete", "abcdef.1234567890123456", "--dry-run"},
			configToWrite: testConfigToken,
			kubeConfigEnv: fullPath,
			expectedError: false,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// the command is created for each test so that the kubeConfigFile
			// variable in NewCmdToken() is reset.
			cmd := NewCmdToken(&buf, &bufErr)
			if _, err = f.WriteString(tc.configToWrite); err != nil {
				t.Errorf("Unable to write test file %q: %v", fullPath, err)
			}
			// store the current value of the environment variable.
			storedEnv := os.Getenv(clientcmd.RecommendedConfigPathEnvVar)
			if tc.kubeConfigEnv != "" {
				os.Setenv(clientcmd.RecommendedConfigPathEnvVar, tc.kubeConfigEnv)
			}
			cmd.SetArgs(tc.args)
			err := cmd.Execute()
			if (err != nil) != tc.expectedError {
				t.Errorf("Test case %q: NewCmdToken expected error: %v, saw: %v", tc.name, tc.expectedError, (err != nil))
			}
			// restore the environment variable.
			os.Setenv(clientcmd.RecommendedConfigPathEnvVar, storedEnv)
		})
	}
}

func TestGetClientset(t *testing.T) {
	testConfigTokenFile := "test-config-file"

	tmpDir, err := ioutil.TempDir("", "kubeadm-token-test")
	if err != nil {
		t.Errorf("Unable to create temporary directory: %v", err)
	}
	defer os.RemoveAll(tmpDir)
	fullPath := filepath.Join(tmpDir, testConfigTokenFile)

	// test dryRun = false on a non-exisiting file
	if _, err = getClientset(fullPath, false); err == nil {
		t.Errorf("getClientset(); dry-run: false; did no fail for test file %q: %v", fullPath, err)
	}

	// test dryRun = true on a non-exisiting file
	if _, err = getClientset(fullPath, true); err == nil {
		t.Errorf("getClientset(); dry-run: true; did no fail for test file %q: %v", fullPath, err)
	}

	f, err := os.Create(fullPath)
	if err != nil {
		t.Errorf("Unable to create test file %q: %v", fullPath, err)
	}
	defer f.Close()

	if _, err = f.WriteString(testConfigToken); err != nil {
		t.Errorf("Unable to write test file %q: %v", fullPath, err)
	}

	// test dryRun = true on an exisiting file
	if _, err = getClientset(fullPath, true); err != nil {
		t.Errorf("getClientset(); dry-run: true; failed for test file %q: %v", fullPath, err)
	}
}

func TestRunDeleteTokens(t *testing.T) {
	var buf bytes.Buffer

	tmpDir, err := ioutil.TempDir("", "kubeadm-token-test")
	if err != nil {
		t.Errorf("Unable to create temporary directory: %v", err)
	}
	defer os.RemoveAll(tmpDir)
	fullPath := filepath.Join(tmpDir, "test-config-file")

	f, err := os.Create(fullPath)
	if err != nil {
		t.Errorf("Unable to create test file %q: %v", fullPath, err)
	}
	defer f.Close()

	if _, err = f.WriteString(testConfigToken); err != nil {
		t.Errorf("Unable to write test file %q: %v", fullPath, err)
	}

	client, err := getClientset(fullPath, true)
	if err != nil {
		t.Errorf("Unable to run getClientset() for test file %q: %v", fullPath, err)
	}

	// test valid; should not fail
	// for some reason Secrets().Delete() does not fail even for this dummy config
	if err = RunDeleteTokens(&buf, client, []string{"abcdef.1234567890123456", "abcdef.2345678901234567"}); err != nil {
		t.Errorf("RunDeleteToken() failed for a valid token: %v", err)
	}

	// test invalid token; should fail
	if err = RunDeleteTokens(&buf, client, []string{"invalid-token"}); err == nil {
		t.Errorf("RunDeleteToken() succeeded for an invalid token: %v", err)
	}
}
