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

package upgrade

import (
	"fmt"
	"io"
	"os"
	"testing"

	"github.com/pkg/errors"

	clientset "k8s.io/client-go/kubernetes"

	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeadmapiv1 "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/v1beta4"
	"k8s.io/kubernetes/cmd/kubeadm/app/util/output"
)

func createTestRunDiffFile(contents []byte) (string, error) {
	file, err := os.CreateTemp("", "kubeadm-upgrade-diff-config-*.yaml")
	if err != nil {
		return "", errors.Wrap(err, "failed to create temporary test file")
	}
	if _, err := file.Write(contents); err != nil {
		return "", errors.Wrap(err, "failed to write to temporary test file")
	}
	if err := file.Close(); err != nil {
		return "", errors.Wrap(err, "failed to close temporary test file")
	}
	return file.Name(), nil
}

func fakeFetchInitConfig(client clientset.Interface, printer output.Printer, logPrefix string, newControlPlane, skipComponentConfigs bool) (*kubeadmapi.InitConfiguration, error) {
	return &kubeadmapi.InitConfiguration{
		ClusterConfiguration: kubeadmapi.ClusterConfiguration{
			KubernetesVersion: "v1.0.1",
		},
	}, nil
}

func TestRunDiff(t *testing.T) {
	// create a temporary file with valid ClusterConfiguration
	testUpgradeDiffConfigContents := []byte(fmt.Sprintf(`
apiVersion: %s
kind: UpgradeConfiguration
diff:
  contextLines: 4`, kubeadmapiv1.SchemeGroupVersion.String()))

	testUpgradeDiffConfig, err := createTestRunDiffFile(testUpgradeDiffConfigContents)
	if err != nil {
		t.Fatal(err)
	}
	defer os.Remove(testUpgradeDiffConfig)
	// create a temporary manifest file with dummy contents
	testUpgradeDiffManifestContents := []byte("some-contents")
	testUpgradeDiffManifest, err := createTestRunDiffFile(testUpgradeDiffManifestContents)
	if err != nil {
		t.Fatal(err)
	}
	defer os.Remove(testUpgradeDiffManifest)

	kubeConfigPath, err := createTestRunDiffFile([]byte(testConfigToken))
	if err != nil {
		t.Fatal(err)
	}
	//nolint:errcheck
	defer os.Remove(kubeConfigPath)

	flags := &diffFlags{
		cfgPath: "",
		out:     io.Discard,
	}

	testCases := []struct {
		name            string
		args            []string
		setManifestPath bool
		manifestPath    string
		cfgPath         string
		expectedError   bool
	}{
		{
			name:            "valid: run diff with empty config path on valid manifest path",
			cfgPath:         "",
			setManifestPath: true,
			manifestPath:    testUpgradeDiffManifest,
			expectedError:   false,
		},
		{
			name:            "valid: run diff on valid manifest path",
			cfgPath:         testUpgradeDiffConfig,
			setManifestPath: true,
			manifestPath:    testUpgradeDiffManifest,
			expectedError:   false,
		},
		{
			name:          "invalid: missing config file",
			cfgPath:       "missing-path-to-a-config",
			expectedError: true,
		},
		{
			name:            "invalid: valid config but empty manifest path",
			cfgPath:         testUpgradeDiffConfig,
			setManifestPath: true,
			manifestPath:    "",
			expectedError:   true,
		},
		{
			name:            "invalid: valid config but bad manifest path",
			cfgPath:         testUpgradeDiffConfig,
			setManifestPath: true,
			manifestPath:    "bad-path",
			expectedError:   true,
		},
		{
			name:          "invalid: badly formatted version as argument",
			cfgPath:       testUpgradeDiffConfig,
			args:          []string{"bad-version"},
			expectedError: true,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			flags.cfgPath = tc.cfgPath
			flags.kubeConfigPath = kubeConfigPath
			cmd := newCmdDiff(os.Stdout)
			if tc.setManifestPath {
				flags.apiServerManifestPath = tc.manifestPath
				flags.controllerManagerManifestPath = tc.manifestPath
				flags.schedulerManifestPath = tc.manifestPath
			}
			if err := runDiff(cmd.Flags(), flags, tc.args, fakeFetchInitConfig); (err != nil) != tc.expectedError {
				t.Fatalf("expected error: %v, saw: %v, error: %v", tc.expectedError, (err != nil), err)
			}
		})
	}
}

func TestValidateManifests(t *testing.T) {
	// Create valid manifest paths
	apiServerManifest, err := createTestRunDiffFile([]byte{})
	if err != nil {
		t.Fatal(err)
	}
	defer os.Remove(apiServerManifest)
	controllerManagerManifest, err := createTestRunDiffFile([]byte{})
	if err != nil {
		t.Fatal(err)
	}
	defer os.Remove(controllerManagerManifest)
	schedulerManifest, err := createTestRunDiffFile([]byte{})
	if err != nil {
		t.Fatal(err)
	}
	defer os.Remove(schedulerManifest)
	// Create a file path that does not exist
	notExistFilePath := "./foobar123456"

	testCases := []struct {
		name          string
		args          []string
		expectedError bool
	}{
		{
			name:          "valid: valid manifest path",
			args:          []string{apiServerManifest, controllerManagerManifest, schedulerManifest},
			expectedError: false,
		},
		{
			name:          "invalid: one is empty path",
			args:          []string{apiServerManifest, controllerManagerManifest, ""},
			expectedError: true,
		},
		{
			name:          "invalid: manifest path is directory",
			args:          []string{"./"},
			expectedError: true,
		},
		{
			name:          "invalid: manifest path does not exist",
			args:          []string{notExistFilePath},
			expectedError: true,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			if err := validateManifestsPath(tc.args...); (err != nil) != tc.expectedError {
				t.Fatalf("expected error: %v, saw: %v, error: %v", tc.expectedError, (err != nil), err)
			}
		})
	}

}
