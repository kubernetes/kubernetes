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
	"bytes"
	"fmt"
	"os"
	"strings"
	"testing"

	clientset "k8s.io/client-go/kubernetes"

	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeadmapiv1 "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/v1beta4"
	"k8s.io/kubernetes/cmd/kubeadm/app/constants"
	"k8s.io/kubernetes/cmd/kubeadm/app/util/errors"
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

func fakeFetchInitConfig(client clientset.Interface, printer output.Printer, logPrefix string, getNodeRegistration, getAPIEndpoint, getComponentConfigs, shortConfigMapGet bool) (*kubeadmapi.InitConfiguration, error) {
	return &kubeadmapi.InitConfiguration{
		ClusterConfiguration: kubeadmapi.ClusterConfiguration{
			KubernetesVersion: "v1.0.1",
		},
	}, nil
}

func fakeFetchInitConfigWithLocalEtcd(client clientset.Interface, printer output.Printer, logPrefix string, getNodeRegistration, getAPIEndpoint, getComponentConfigs, shortConfigMapGet bool) (*kubeadmapi.InitConfiguration, error) {
	return &kubeadmapi.InitConfiguration{
		ClusterConfiguration: kubeadmapi.ClusterConfiguration{
			KubernetesVersion: "v1.0.1",
			Etcd: kubeadmapi.Etcd{
				Local: &kubeadmapi.LocalEtcd{
					ImageMeta: kubeadmapi.ImageMeta{ImageTag: "3.5.0-0"},
					DataDir:   "/var/lib/etcd",
				},
			},
		},
	}, nil
}

func createTestRunDiffEtcdManifest(t *testing.T, imageTag string) string {
	manifest := fmt.Sprintf(`apiVersion: v1
kind: Pod
metadata:
  name: etcd
  namespace: kube-system
spec:
  containers:
  - name: etcd
    image: registry.k8s.io/etcd:%s
`, imageTag)
	path := constants.GetStaticPodFilepath(constants.Etcd, t.TempDir())
	if err := os.WriteFile(path, []byte(manifest), 0644); err != nil {
		t.Fatalf("failed to write test etcd manifest: %v", err)
	}
	return path
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

	noVersionEtcdManifest := createTestRunDiffEtcdManifest(t, "latest")
	// etcd manifests on either side of the desired tag of fakeFetchInitConfigWithLocalEtcd
	olderEtcdManifest := createTestRunDiffEtcdManifest(t, "3.4.0-0")
	newerEtcdManifest := createTestRunDiffEtcdManifest(t, "3.6.0-0")

	flags := &diffFlags{
		cfgPath: "",
	}

	testCases := []struct {
		name             string
		args             []string
		setManifestPath  bool
		manifestPath     string
		etcdManifestPath string
		fetchInitConfig  FetchInitConfigurationFunc
		cfgPath          string
		expectedError    bool
		expectedEtcdDiff bool
	}{
		{
			name:            "valid: run diff with empty config path on valid manifest path",
			cfgPath:         "",
			setManifestPath: true,
			manifestPath:    testUpgradeDiffManifest,
			expectedError:   false,
		},
		{
			name:             "valid: run diff with local etcd on valid manifest paths",
			cfgPath:          testUpgradeDiffConfig,
			setManifestPath:  true,
			manifestPath:     testUpgradeDiffManifest,
			etcdManifestPath: olderEtcdManifest,
			fetchInitConfig:  fakeFetchInitConfigWithLocalEtcd,
			expectedError:    false,
			expectedEtcdDiff: true,
		},
		{
			name:             "valid: local etcd newer than the desired version is left out of the diff",
			cfgPath:          testUpgradeDiffConfig,
			setManifestPath:  true,
			manifestPath:     testUpgradeDiffManifest,
			etcdManifestPath: newerEtcdManifest,
			fetchInitConfig:  fakeFetchInitConfigWithLocalEtcd,
			expectedError:    false,
			expectedEtcdDiff: false,
		},
		{
			name:             "invalid: local etcd manifest without a version in its image tag",
			cfgPath:          testUpgradeDiffConfig,
			setManifestPath:  true,
			manifestPath:     testUpgradeDiffManifest,
			etcdManifestPath: noVersionEtcdManifest,
			fetchInitConfig:  fakeFetchInitConfigWithLocalEtcd,
			expectedError:    true,
		},
		{
			name:             "invalid: local etcd but bad etcd manifest path",
			cfgPath:          testUpgradeDiffConfig,
			setManifestPath:  true,
			manifestPath:     testUpgradeDiffManifest,
			etcdManifestPath: "bad-path",
			fetchInitConfig:  fakeFetchInitConfigWithLocalEtcd,
			expectedError:    true,
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
			flags.etcdManifestPath = tc.etcdManifestPath
			out := &bytes.Buffer{}
			flags.out = out
			fetchInitConfig := tc.fetchInitConfig
			if fetchInitConfig == nil {
				fetchInitConfig = fakeFetchInitConfig
			}
			if err := runDiff(cmd.Flags(), flags, tc.args, fetchInitConfig); (err != nil) != tc.expectedError {
				t.Fatalf("expected error: %v, saw: %v, error: %v", tc.expectedError, (err != nil), err)
			}
			if etcdDiff := strings.Contains(out.String(), "--- "+tc.etcdManifestPath+"\n"); etcdDiff != tc.expectedEtcdDiff {
				t.Fatalf("expected etcd manifest in the diff: %v, saw: %v, output:\n%s", tc.expectedEtcdDiff, etcdDiff, out.String())
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
