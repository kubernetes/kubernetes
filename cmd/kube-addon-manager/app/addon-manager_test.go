/*
Copyright 2020 The Kubernetes Authors.

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

package app

import (
	"errors"
	"io"
	"io/ioutil"
	"os"
	"path/filepath"
	"reflect"
	"testing"
	"time"
)

func TestIsLeader(t *testing.T) {
	testHostname := "test-instance"
	testKubectlOpts := "--kubeconfig=/some/dir/kubeconfig.json"
	expectedKubectlArgs := []string{testKubectlOpts, "-n", "kube-system", "get", "ep", "kube-controller-manager", "-o", `go-template={{index .metadata.annotations "control-plane.alpha.kubernetes.io/leader"}}`}
	tcs := []struct {
		desc          string
		kubectlOut    string
		kubectlErrOut string
		kubectlErr    error

		expected bool
	}{
		{
			desc:       "happy case, is leader",
			kubectlOut: `{"holderIdentity":"test-instance_a6d5dd56-c545-11ea-b852-42010a800240","leaseDurationSeconds":15,"acquireTime":"2020-07-13T20:16:19Z","renewTime":"2020-07-16T16:48:51Z","leaderTransitions":0}`,
			expected:   true,
		},
		{
			desc:       "happy case, isn't leader",
			kubectlOut: `{"holderIdentity":"some-other-instance_some-uuid","leaseDurationSeconds":15,"acquireTime":"2020-07-13T20:16:19Z","renewTime":"2020-07-16T16:48:51Z","leaderTransitions":0}`,
			expected:   false,
		},
		{
			desc:       "holderIdentity empty",
			kubectlOut: `{"holderIdentity":"","leaseDurationSeconds":15,"acquireTime":"2020-07-13T20:16:19Z","renewTime":"2020-07-16T16:48:51Z","leaderTransitions":0}`,
			expected:   true,
		},
		{
			desc:       "no holderIdentity",
			kubectlOut: `{"leaseDurationSeconds":15,"acquireTime":"2020-07-13T20:16:19Z","renewTime":"2020-07-16T16:48:51Z","leaderTransitions":0}`,
			expected:   true,
		},
		{
			desc:       "empty output",
			kubectlOut: "",
			expected:   true,
		},
		{
			desc:          "kubectl error",
			kubectlErrOut: "not found",
			kubectlErr:    errors.New("exit code: 1"),
			expected:      true,
		},
	}

	for _, tc := range tcs {
		t.Run(tc.desc, func(t *testing.T) {
			am := &addonManager{
				hostname:    testHostname,
				kubectlOpts: testKubectlOpts,
				kubectl: func(stdout, stderr io.Writer, args ...string) error {
					if !reflect.DeepEqual(expectedKubectlArgs, args) {
						t.Errorf("incorrect args passed to kubectl. Wanted: %#v, Got: %#v", expectedKubectlArgs, args)
					}
					if tc.kubectlOut != "" {
						io.WriteString(stdout, tc.kubectlOut)
					}
					if tc.kubectlErrOut != "" {
						io.WriteString(stderr, tc.kubectlErrOut)
					}
					return tc.kubectlErr
				},
			}
			result := am.isLeader()
			if tc.expected != result {
				t.Errorf("isLeader() returned %v, wanted %v", result, tc.expected)
			}
		})
	}
}

func TestNewAddonManager(t *testing.T) {
	tcs := []struct {
		desc string
		env  map[string]string

		expected *addonManager
	}{
		{
			desc: "default",
			env:  map[string]string{},

			expected: &addonManager{
				addonPath:            "/etc/kubernetes/addons",
				admissionControlsDir: "/etc/kubernetes/admission-controls",
				checkInterval:        60 * time.Second,
				kubectlPath:          "/usr/local/bin/kubectl",
				leaderElection:       true,
				pruneWhitelistFlags: []string{
					"--prune-whitelist", "core/v1/ConfigMap",
					"--prune-whitelist", "core/v1/Endpoints",
					"--prune-whitelist", "core/v1/Namespace",
					"--prune-whitelist", "core/v1/PersistentVolumeClaim",
					"--prune-whitelist", "core/v1/PersistentVolume",
					"--prune-whitelist", "core/v1/Pod",
					"--prune-whitelist", "core/v1/ReplicationController",
					"--prune-whitelist", "core/v1/Secret",
					"--prune-whitelist", "core/v1/Service",
					"--prune-whitelist", "batch/v1/Job",
					"--prune-whitelist", "batch/v1beta1/CronJob",
					"--prune-whitelist", "apps/v1/DaemonSet",
					"--prune-whitelist", "apps/v1/Deployment",
					"--prune-whitelist", "apps/v1/ReplicaSet",
					"--prune-whitelist", "apps/v1/StatefulSet",
					"--prune-whitelist", "extensions/v1beta1/Ingress",
				},
			},
		},
		{
			desc: "all env vars set",
			env: map[string]string{
				"HOSTNAME":                         "testHostname",
				"KUBECTL_OPTS":                     "--kubeconfig=/some/dir/kubeconfig.json",
				"ADDON_PATH":                       "/test/addon/dir",
				"KUBECTL_BIN":                      "/usr/bin/testKubectl",
				"ADDON_MANAGER_LEADER_ELECTION":    "false",
				"KUBECTL_PRUNE_WHITELIST_OVERRIDE": "core/v1/OverrideResource extensions/v1beta1/OverrideResource",
				"KUBECTL_EXTRA_PRUNE_WHITELIST":    "core/v1/ExtraResource extensions/v1beta1/ExtraResource",
				"TEST_ADDON_CHECK_INTERVAL_SEC":    "42",
			},

			expected: &addonManager{
				addonPath:            "/test/addon/dir",
				admissionControlsDir: "/etc/kubernetes/admission-controls",
				checkInterval:        42 * time.Second,
				hostname:             "testHostname",
				kubectlOpts:          "--kubeconfig=/some/dir/kubeconfig.json",
				kubectlPath:          "/usr/bin/testKubectl",
				leaderElection:       false,
				pruneWhitelistFlags: []string{
					"--prune-whitelist", "core/v1/OverrideResource",
					"--prune-whitelist", "extensions/v1beta1/OverrideResource",
					"--prune-whitelist", "core/v1/ExtraResource",
					"--prune-whitelist", "extensions/v1beta1/ExtraResource",
				},
			},
		},
	}

	for _, tc := range tcs {
		t.Run(tc.desc, func(t *testing.T) {
			am, err := newAddonManager(func(key string) string {
				return tc.env[key]
			})
			if err != nil {
				t.Fatalf("newAddonManager() returned error: %v", err)
			}
			if am.kubectl == nil {
				t.Error("newAddonManager() returned an addon manager without `kubectl` implementation.")
			}
			am.kubectl = nil // set function var to nil so that DeepEqual can work
			if !reflect.DeepEqual(tc.expected, am) {
				t.Errorf("newAddonManager() returned %#v, wanted %#v", tc.expected, am)
			}
		})
	}
}

func TestMakePruneWhitelistFlags(t *testing.T) {
	tcs := []struct {
		desc                   string
		pruneWhitelistOverride string
		extraPruneWhitelist    string

		expected []string
	}{
		{
			desc: "default",

			expected: []string{
				"--prune-whitelist", "core/v1/ConfigMap",
				"--prune-whitelist", "core/v1/Endpoints",
				"--prune-whitelist", "core/v1/Namespace",
				"--prune-whitelist", "core/v1/PersistentVolumeClaim",
				"--prune-whitelist", "core/v1/PersistentVolume",
				"--prune-whitelist", "core/v1/Pod",
				"--prune-whitelist", "core/v1/ReplicationController",
				"--prune-whitelist", "core/v1/Secret",
				"--prune-whitelist", "core/v1/Service",
				"--prune-whitelist", "batch/v1/Job",
				"--prune-whitelist", "batch/v1beta1/CronJob",
				"--prune-whitelist", "apps/v1/DaemonSet",
				"--prune-whitelist", "apps/v1/Deployment",
				"--prune-whitelist", "apps/v1/ReplicaSet",
				"--prune-whitelist", "apps/v1/StatefulSet",
				"--prune-whitelist", "extensions/v1beta1/Ingress",
			},
		},
		{
			desc:                   "override set",
			pruneWhitelistOverride: "core/v1/OverrideResource extensions/v1beta1/OverrideResource",

			expected: []string{
				"--prune-whitelist", "core/v1/OverrideResource",
				"--prune-whitelist", "extensions/v1beta1/OverrideResource",
			},
		},
		{
			desc:                "extra set",
			extraPruneWhitelist: "core/v1/ExtraResource extensions/v1beta1/ExtraResource",

			expected: []string{
				"--prune-whitelist", "core/v1/ConfigMap",
				"--prune-whitelist", "core/v1/Endpoints",
				"--prune-whitelist", "core/v1/Namespace",
				"--prune-whitelist", "core/v1/PersistentVolumeClaim",
				"--prune-whitelist", "core/v1/PersistentVolume",
				"--prune-whitelist", "core/v1/Pod",
				"--prune-whitelist", "core/v1/ReplicationController",
				"--prune-whitelist", "core/v1/Secret",
				"--prune-whitelist", "core/v1/Service",
				"--prune-whitelist", "batch/v1/Job",
				"--prune-whitelist", "batch/v1beta1/CronJob",
				"--prune-whitelist", "apps/v1/DaemonSet",
				"--prune-whitelist", "apps/v1/Deployment",
				"--prune-whitelist", "apps/v1/ReplicaSet",
				"--prune-whitelist", "apps/v1/StatefulSet",
				"--prune-whitelist", "extensions/v1beta1/Ingress",
				"--prune-whitelist", "core/v1/ExtraResource",
				"--prune-whitelist", "extensions/v1beta1/ExtraResource",
			},
		},
		{
			desc:                   "override and extra set",
			pruneWhitelistOverride: "core/v1/OverrideResource extensions/v1beta1/OverrideResource",
			extraPruneWhitelist:    "core/v1/ExtraResource extensions/v1beta1/ExtraResource",

			expected: []string{
				"--prune-whitelist", "core/v1/OverrideResource",
				"--prune-whitelist", "extensions/v1beta1/OverrideResource",
				"--prune-whitelist", "core/v1/ExtraResource",
				"--prune-whitelist", "extensions/v1beta1/ExtraResource",
			},
		},
	}

	for _, tc := range tcs {
		t.Run(tc.desc, func(t *testing.T) {
			flags := makePruneWhitelistFlags(func(key string) string {
				if key == "KUBECTL_PRUNE_WHITELIST_OVERRIDE" {
					return tc.pruneWhitelistOverride
				}
				if key == "KUBECTL_EXTRA_PRUNE_WHITELIST" {
					return tc.extraPruneWhitelist
				}
				t.Errorf("unknown ENV value requested: %q", key)
				return ""
			})
			if !reflect.DeepEqual(tc.expected, flags) {
				t.Errorf("makePruneWhitelistFlags() returned %v, wanted %v", flags, tc.expected)
			}
		})
	}
}

func TestWaitForSystemServiceAccount(t *testing.T) {
	t.Parallel()
	testHostname := "test-instance"
	testKubectlOpts := "--kubeconfig=/some/dir/kubeconfig.json"
	expectedKubectlArgs := []string{testKubectlOpts, "get", "-n", "kube-system", "serviceaccount", "default", "-o", "go-template={{with index .secrets 0}}{{.name}}{{end}}"}
	tcs := []struct {
		desc        string
		kubectlOuts []string
		kubectlErrs []error

		expectedAttempts int
	}{
		{
			desc:             "happy case",
			kubectlOuts:      []string{"service-account-t0ken"},
			expectedAttempts: 1,
		},
		{
			desc:             "wait for token",
			kubectlOuts:      []string{"", "", "service-account-t0ken"},
			expectedAttempts: 3,
		},
		{
			desc:             "retry on error",
			kubectlOuts:      []string{"", "service-account-t0ken"},
			kubectlErrs:      []error{errors.New("exit status: 1")},
			expectedAttempts: 2,
		},
	}

	for _, tc := range tcs {
		tc := tc
		t.Run(tc.desc, func(t *testing.T) {
			t.Parallel()
			numAttempts := 0
			am := &addonManager{
				hostname:    testHostname,
				kubectlOpts: testKubectlOpts,
				kubectl: func(stdout, stderr io.Writer, args ...string) error {
					numAttempts++
					if !reflect.DeepEqual(expectedKubectlArgs, args) {
						t.Errorf("incorrect args passed to kubectl. Wanted: %#v, Got: %#v", expectedKubectlArgs, args)
					}
					if len(tc.kubectlOuts) > 0 {
						io.WriteString(stdout, tc.kubectlOuts[0])
						tc.kubectlOuts = tc.kubectlOuts[1:]
					}
					if len(tc.kubectlErrs) > 0 {
						err := tc.kubectlErrs[0]
						tc.kubectlErrs = tc.kubectlErrs[1:]
						return err
					}
					return nil
				},
			}
			am.waitForSystemServiceAccount()
			if tc.expectedAttempts != numAttempts {
				t.Errorf("waitForSystemServiceAccount() made %d attempts to wait for the kube-system service account, expected %d", numAttempts, tc.expectedAttempts)
			}
		})
	}
}

func TestReconcileAddons(t *testing.T) {
	testKubectlOpts := "--kubeconfig=/some/dir/kubeconfig.json"
	testAddonPath := "/test/addon/dir"
	testPruneWhitelistFlags := []string{"--prune-whitelist", "core/v1/ReconcileAddonsTestResource"}
	expectedDeprecatedLabelKubectlArgs := []string{
		testKubectlOpts,
		"apply", "-f", testAddonPath, "--recursive",
		"-l", "kubernetes.io/cluster-service=true,addonmanager.kubernetes.io/mode!=EnsureExists",
		"--prune=true", "--prune-whitelist", "core/v1/ReconcileAddonsTestResource"}
	expectedKubectlArgs := []string{
		testKubectlOpts,
		"apply", "-f", testAddonPath, "--recursive",
		"-l", "kubernetes.io/cluster-service!=true,addonmanager.kubernetes.io/mode=Reconcile",
		"--prune=true", "--prune-whitelist", "core/v1/ReconcileAddonsTestResource"}

	var calledWithDeprecatedLabel, calledWithAddonManagerLabel bool

	am := &addonManager{
		addonPath:           testAddonPath,
		kubectlOpts:         testKubectlOpts,
		pruneWhitelistFlags: testPruneWhitelistFlags,
		kubectl: func(_, _ io.Writer, args ...string) error {
			if reflect.DeepEqual(expectedDeprecatedLabelKubectlArgs, args) {
				calledWithDeprecatedLabel = true
				return nil
			}
			if reflect.DeepEqual(expectedKubectlArgs, args) {
				calledWithAddonManagerLabel = true
				return nil
			}

			t.Errorf("kubectl called with unexpected args: %#v", args)
			return errors.New("called expectedly")
		},
	}

	am.reconcileAddons()

	if !calledWithDeprecatedLabel {
		t.Error("expected `kubectl apply` to be called with deprecated label, was not")
	}
	if !calledWithAddonManagerLabel {
		t.Error("expected `kubectl apply` to be called with addon-manager label, was not")
	}
}

func TestEnsureAddons(t *testing.T) {
	testKubectlOpts := "--kubeconfig=/some/dir/kubeconfig.json"
	testAddonPath := "/test/addon/dir"
	testPruneWhitelistFlags := []string{"--prune-whitelist", "core/v1/ReconcileAddonsTestResource"}
	expectedKubectlArgs := []string{
		testKubectlOpts,
		"create", "-f", "/test/addon/dir", "--recursive",
		"-l", "addonmanager.kubernetes.io/mode=EnsureExists"}

	var called bool

	am := &addonManager{
		addonPath:           testAddonPath,
		kubectlOpts:         testKubectlOpts,
		pruneWhitelistFlags: testPruneWhitelistFlags,
		kubectl: func(_, _ io.Writer, args ...string) error {
			if reflect.DeepEqual(expectedKubectlArgs, args) {
				called = true
				return nil
			}

			t.Errorf("kubectl called with unexpected args: %#v", args)
			return errors.New("called expectedly")
		},
	}

	am.ensureAddons()

	if !called {
		t.Error("expected `kubectl create` to be called with expected args, was not")
	}
}

func TestExtractHolderIdentity(t *testing.T) {
	tcs := []struct {
		desc  string
		input string

		expected string
	}{
		{
			desc:     "happy case",
			input:    `{"holderIdentity":"test-instance","leaseDurationSeconds":15,"acquireTime":"2020-07-13T20:16:19Z","renewTime":"2020-07-16T16:48:51Z","leaderTransitions":0}`,
			expected: "test-instance",
		},
		{
			desc:     "holderIdentity empty",
			input:    `{"holderIdentity":"","leaseDurationSeconds":15,"acquireTime":"2020-07-13T20:16:19Z","renewTime":"2020-07-16T16:48:51Z","leaderTransitions":0}`,
			expected: "",
		},
		{
			desc:     "no holderIdentity",
			input:    `{"leaseDurationSeconds":15,"acquireTime":"2020-07-13T20:16:19Z","renewTime":"2020-07-16T16:48:51Z","leaderTransitions":0}`,
			expected: "",
		},
		{
			desc:     "empty input",
			input:    "",
			expected: "",
		},
	}

	for _, tc := range tcs {
		t.Run(tc.desc, func(t *testing.T) {
			result := extractHolderIdentity(tc.input)
			if tc.expected != result {
				t.Errorf("extractHolderIdentity(%q) returned %q, wanted %q", tc.input, result, tc.expected)
			}
		})
	}
}

func TestEnsureDefaultAdmissionControlsObjects(t *testing.T) {
	testKubectlOpts := "--kubeconfig=/some/dir/kubeconfig.json"

	tcs := []struct {
		desc           string
		testFiles      []string
		extraTestFiles []string
	}{
		{
			desc:      "json",
			testFiles: []string{"test_file.json"},
		},
		{
			desc:      "yaml",
			testFiles: []string{"test_file.yaml"},
		},
		{
			desc:           "mixed",
			testFiles:      []string{"test_file.yaml", "test_file.json"},
			extraTestFiles: []string{"should_be_ignored.txt"},
		},
		{
			desc:           "nested dir",
			testFiles:      []string{"test_file.yaml", "nested/test_file.json"},
			extraTestFiles: []string{"should_be_ignored.txt"},
		},
	}

	for _, tc := range tcs {
		t.Run(tc.desc, func(t *testing.T) {
			testDir, err := ioutil.TempDir("", "test_admissions_controls_dir_*")
			if err != nil {
				t.Fatalf("error creating test directory: %v", err)
			}
			defer os.RemoveAll(testDir)

			for _, testFile := range tc.testFiles {
				path := filepath.Join(testDir, testFile)
				os.MkdirAll(filepath.Dir(path), os.ModePerm)
				if err := ioutil.WriteFile(path, []byte("test file contents"), 0666); err != nil {
					t.Fatalf("error creating test file %s: %v", path, err)
				}
			}
			for _, extraFile := range tc.extraTestFiles {
				path := filepath.Join(testDir, extraFile)
				os.MkdirAll(filepath.Dir(path), os.ModePerm)
				if err := ioutil.WriteFile(path, []byte("extra test file contents"), 0666); err != nil {
					t.Fatalf("error creating extra test file %s: %v", path, err)
				}
			}

			kubectlArgs := [][]string{}
			am := &addonManager{
				admissionControlsDir: testDir,
				kubectlOpts:          testKubectlOpts,
				kubectl: func(_, _ io.Writer, args ...string) error {
					kubectlArgs = append(kubectlArgs, args)
					return nil
				},
			}

			am.ensureDefaultAdmissionControlsObjects()

			for _, testFile := range tc.testFiles {
				expectedArgs := []string{testKubectlOpts, "--namespace=default", "apply", "-f", filepath.Join(testDir, testFile)}
				call := -1
				for i, args := range kubectlArgs {
					if reflect.DeepEqual(args, expectedArgs) {
						call = i
						break
					}
				}
				if call == -1 {
					t.Errorf("expected `kubectl` to be called with args %q", expectedArgs)
					continue
				}
				kubectlArgs = append(kubectlArgs[:call], kubectlArgs[call+1:]...)
			}
			for _, args := range kubectlArgs {
				t.Errorf("`kubectl` unexpectedly called with args %q", args)
			}
		})
	}

}
