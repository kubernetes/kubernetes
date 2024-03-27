/*
Copyright 2022 The Kubernetes Authors.

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
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/sets"

	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeadmapiv1 "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/v1beta4"
	"k8s.io/kubernetes/cmd/kubeadm/app/cmd/options"
	"k8s.io/kubernetes/cmd/kubeadm/app/constants"
)

var defaultURLScheme = kubeadmapiv1.DefaultContainerRuntimeURLScheme
var testResetConfig = fmt.Sprintf(`apiVersion: %s
kind: ResetConfiguration
force: true
dryRun: true
cleanupTmpDir: true
criSocket: %s:///var/run/fake.sock
certificatesDir: /etc/kubernetes/pki2
ignorePreflightErrors:
- a
- b
`, kubeadmapiv1.SchemeGroupVersion.String(), defaultURLScheme)

func TestNewResetData(t *testing.T) {
	// create temp directory
	tmpDir, err := os.MkdirTemp("", "kubeadm-reset-test")
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
	if _, err = cfgFile.WriteString(testResetConfig); err != nil {
		t.Fatalf("Unable to write file %q: %v", configFilePath, err)
	}

	testCases := []struct {
		name        string
		args        []string
		flags       map[string]string
		validate    func(*testing.T, *resetData)
		expectError string
		data        *resetData
	}{
		{
			name: "flags parsed correctly",
			flags: map[string]string{
				options.CertificatesDir:       "/tmp",
				options.NodeCRISocket:         constants.CRISocketCRIO,
				options.IgnorePreflightErrors: "all",
				options.ForceReset:            "true",
				options.DryRun:                "true",
				options.CleanupTmpDir:         "true",
			},
			data: &resetData{
				certificatesDir:       "/tmp",
				criSocketPath:         constants.CRISocketCRIO,
				ignorePreflightErrors: sets.New("all"),
				forceReset:            true,
				dryRun:                true,
				cleanupTmpDir:         true,
				// resetCfg holds the value passed from flags except the value of ignorePreflightErrors
				resetCfg: &kubeadmapi.ResetConfiguration{
					TypeMeta:        metav1.TypeMeta{Kind: "", APIVersion: ""},
					Force:           true,
					CertificatesDir: "/tmp",
					CRISocket:       constants.CRISocketCRIO,
					DryRun:          true,
					CleanupTmpDir:   true,
				},
			},
		},
		{
			name: "fails if preflight ignores all but has individual check",
			flags: map[string]string{
				options.IgnorePreflightErrors: "all,something-else",
				options.NodeCRISocket:         fmt.Sprintf("%s:///var/run/crio/crio.sock", defaultURLScheme),
			},
			expectError: "don't specify individual checks if 'all' is used",
		},
		{
			name: "pre-flights errors from CLI args",
			flags: map[string]string{
				options.IgnorePreflightErrors: "a,b",
				options.NodeCRISocket:         fmt.Sprintf("%s:///var/run/crio/crio.sock", defaultURLScheme),
			},
			validate: expectedResetIgnorePreflightErrors(sets.New("a", "b")),
		},
		// Start the testcases with config file
		{
			name: "Pass with config from file",
			flags: map[string]string{
				options.CfgPath: configFilePath,
			},
			data: &resetData{
				certificatesDir:       "/etc/kubernetes/pki2",                                   // cover the case that default is overridden as well
				criSocketPath:         fmt.Sprintf("%s:///var/run/fake.sock", defaultURLScheme), // cover the case that default is overridden as well
				ignorePreflightErrors: sets.New("a", "b"),
				forceReset:            true,
				dryRun:                true,
				cleanupTmpDir:         true,
				resetCfg: &kubeadmapi.ResetConfiguration{
					TypeMeta:              metav1.TypeMeta{Kind: "", APIVersion: ""},
					Force:                 true,
					CertificatesDir:       "/etc/kubernetes/pki2",
					CRISocket:             fmt.Sprintf("%s:///var/run/fake.sock", defaultURLScheme),
					IgnorePreflightErrors: []string{"a", "b"},
					CleanupTmpDir:         true,
					DryRun:                true,
				},
			},
		},
		{
			name: "force from config file overrides default",
			flags: map[string]string{
				options.CfgPath: configFilePath,
			},
			validate: func(t *testing.T, data *resetData) {
				// validate that the default value is overwritten
				if data.forceReset != true {
					t.Error("Invalid forceReset")
				}
			},
		},
		{
			name: "dryRun configured in the config file only",
			flags: map[string]string{
				options.CfgPath: configFilePath,
			},
			validate: func(t *testing.T, data *resetData) {
				if data.dryRun != true {
					t.Error("Invalid dryRun")
				}
			},
		},
		{
			name: "--cert-dir flag is not allowed to mix with config",
			flags: map[string]string{
				options.CfgPath:         configFilePath,
				options.CertificatesDir: "/tmp",
			},
			expectError: "can not mix '--config' with arguments",
		},
		{
			name: "--cri-socket flag is not allowed to mix with config",
			flags: map[string]string{
				options.CfgPath:       configFilePath,
				options.NodeCRISocket: fmt.Sprintf("%s:///var/run/bogus.sock", defaultURLScheme),
			},
			expectError: "can not mix '--config' with arguments",
		},
		{
			name: "--force flag is not allowed to mix with config",
			flags: map[string]string{
				options.CfgPath:    configFilePath,
				options.ForceReset: "false",
			},
			expectError: "can not mix '--config' with arguments",
		},
		{
			name: "--cleanup-tmp-dir flag is not allowed to mix with config",
			flags: map[string]string{
				options.CfgPath:       configFilePath,
				options.CleanupTmpDir: "true",
			},
			expectError: "can not mix '--config' with arguments",
		},
		{
			name: "pre-flights errors from ResetConfiguration only",
			flags: map[string]string{
				options.CfgPath: configFilePath,
			},
			validate: expectedResetIgnorePreflightErrors(sets.New("a", "b")),
		},
		{
			name: "pre-flights errors from both CLI args and ResetConfiguration",
			flags: map[string]string{
				options.CfgPath:               configFilePath,
				options.IgnorePreflightErrors: "c,d",
			},
			validate: expectedResetIgnorePreflightErrors(sets.New("a", "b", "c", "d")),
		},
	}
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// initialize an external reset option and inject it to the reset cmd
			resetOptions := newResetOptions()
			cmd := newCmdReset(nil, nil, resetOptions)

			// sets cmd flags (that will be reflected on the reset options)
			for f, v := range tc.flags {
				cmd.Flags().Set(f, v)
			}

			// test newResetData method
			data, err := newResetData(cmd, resetOptions, nil, nil, true)
			if err != nil && !strings.Contains(err.Error(), tc.expectError) {
				t.Fatalf("newResetData returned unexpected error, expected: %s, got %v", tc.expectError, err)
			}
			if err == nil && len(tc.expectError) != 0 {
				t.Fatalf("newResetData didn't return error expected %s", tc.expectError)
			}

			if tc.data != nil {
				if diff := cmp.Diff(tc.data, data, cmp.AllowUnexported(resetData{}), cmpopts.IgnoreFields(resetData{}, "client", "resetCfg.Timeouts")); diff != "" {
					t.Fatalf("newResetData returned data (-want,+got):\n%s", diff)
				}
			}
			// exec additional validation on the returned value
			if tc.validate != nil {
				tc.validate(t, data)
			}
		})
	}
}

func expectedResetIgnorePreflightErrors(expected sets.Set[string]) func(t *testing.T, data *resetData) {
	return func(t *testing.T, data *resetData) {
		if !expected.Equal(data.ignorePreflightErrors) {
			t.Errorf("Invalid ignore preflight errors. Expected: %v. Actual: %v", sets.List(expected), sets.List(data.ignorePreflightErrors))
		}
		if data.cfg != nil && !expected.HasAll(data.cfg.NodeRegistration.IgnorePreflightErrors...) {
			t.Errorf("Invalid ignore preflight errors in InitConfiguration. Expected: %v. Actual: %v", sets.List(expected), data.cfg.NodeRegistration.IgnorePreflightErrors)
		}
	}
}
