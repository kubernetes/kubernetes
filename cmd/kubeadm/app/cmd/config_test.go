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

package cmd_test

import (
	"bytes"
	"io/ioutil"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/renstrom/dedent"

	kubeadmapiv1alpha2 "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/v1alpha2"
	"k8s.io/kubernetes/cmd/kubeadm/app/cmd"
	"k8s.io/kubernetes/cmd/kubeadm/app/features"
	"k8s.io/kubernetes/cmd/kubeadm/app/util/config"
)

const (
	defaultNumberOfImages = 8
	// dummyKubernetesVersion is just used for unit testing, in order to not make
	// kubeadm lookup dl.k8s.io to resolve what the latest stable release is
	dummyKubernetesVersion = "v1.10.0"
)

func TestNewCmdConfigImagesList(t *testing.T) {
	var output bytes.Buffer
	mockK8sVersion := dummyKubernetesVersion
	images := cmd.NewCmdConfigImagesList(&output, &mockK8sVersion)
	images.Run(nil, nil)
	actual := strings.Split(output.String(), "\n")
	if len(actual) != defaultNumberOfImages {
		t.Fatalf("Expected %v but found %v images", defaultNumberOfImages, len(actual))
	}
}

func TestImagesListRunWithCustomConfigPath(t *testing.T) {
	testcases := []struct {
		name               string
		expectedImageCount int
		// each string provided here must appear in at least one image returned by Run
		expectedImageSubstrings []string
		configContents          []byte
	}{
		{
			name:               "set k8s version",
			expectedImageCount: defaultNumberOfImages,
			expectedImageSubstrings: []string{
				":v1.10.1",
			},
			configContents: []byte(dedent.Dedent(`
				apiVersion: kubeadm.k8s.io/v1alpha2
				kind: MasterConfiguration
				kubernetesVersion: v1.10.1
			`)),
		},
		{
			name:               "use coredns",
			expectedImageCount: defaultNumberOfImages,
			expectedImageSubstrings: []string{
				"coredns",
			},
			configContents: []byte(dedent.Dedent(`
				apiVersion: kubeadm.k8s.io/v1alpha2
				kind: MasterConfiguration
				kubernetesVersion: v1.11.0
				featureGates:
				  CoreDNS: True
			`)),
		},
	}

	for _, tc := range testcases {
		t.Run(tc.name, func(t *testing.T) {
			tmpDir, err := ioutil.TempDir("", "kubeadm-images-test")
			if err != nil {
				t.Fatalf("Unable to create temporary directory: %v", err)
			}
			defer os.RemoveAll(tmpDir)

			configFilePath := filepath.Join(tmpDir, "test-config-file")
			err = ioutil.WriteFile(configFilePath, tc.configContents, 0644)
			if err != nil {
				t.Fatalf("Failed writing a config file: %v", err)
			}

			i, err := cmd.NewImagesList(configFilePath, &kubeadmapiv1alpha2.MasterConfiguration{
				KubernetesVersion: dummyKubernetesVersion,
			})
			if err != nil {
				t.Fatalf("Failed getting the kubeadm images command: %v", err)
			}
			var output bytes.Buffer
			if i.Run(&output) != nil {
				t.Fatalf("Error from running the images command: %v", err)
			}
			actual := strings.Split(output.String(), "\n")
			if len(actual) != tc.expectedImageCount {
				t.Fatalf("did not get the same number of images: actual: %v expected: %v. Actual value: %v", len(actual), tc.expectedImageCount, actual)
			}

			for _, substring := range tc.expectedImageSubstrings {
				if !strings.Contains(output.String(), substring) {
					t.Errorf("Expected to find %v but did not in this list of images: %v", substring, actual)
				}
			}
		})
	}
}

func TestConfigImagesListRunWithoutPath(t *testing.T) {
	testcases := []struct {
		name           string
		cfg            kubeadmapiv1alpha2.MasterConfiguration
		expectedImages int
	}{
		{
			name:           "empty config",
			expectedImages: defaultNumberOfImages,
			cfg: kubeadmapiv1alpha2.MasterConfiguration{
				KubernetesVersion: dummyKubernetesVersion,
			},
		},
		{
			name: "external etcd configuration",
			cfg: kubeadmapiv1alpha2.MasterConfiguration{
				Etcd: kubeadmapiv1alpha2.Etcd{
					External: &kubeadmapiv1alpha2.ExternalEtcd{
						Endpoints: []string{"https://some.etcd.com:2379"},
					},
				},
				KubernetesVersion: dummyKubernetesVersion,
			},
			expectedImages: defaultNumberOfImages - 1,
		},
		{
			name: "coredns enabled",
			cfg: kubeadmapiv1alpha2.MasterConfiguration{
				FeatureGates: map[string]bool{
					features.CoreDNS: true,
				},
				KubernetesVersion: dummyKubernetesVersion,
			},
			expectedImages: defaultNumberOfImages,
		},
	}

	for _, tc := range testcases {
		t.Run(tc.name, func(t *testing.T) {
			i, err := cmd.NewImagesList("", &tc.cfg)
			if err != nil {
				t.Fatalf("did not expect an error while creating the Images command: %v", err)
			}

			var output bytes.Buffer
			if i.Run(&output) != nil {
				t.Fatalf("did not expect an error running the Images command: %v", err)
			}

			actual := strings.Split(output.String(), "\n")
			if len(actual) != tc.expectedImages {
				t.Fatalf("expected %v images but got %v", tc.expectedImages, actual)
			}
		})
	}
}

type fakePuller struct {
	count map[string]int
}

func (f *fakePuller) Pull(image string) error {
	f.count[image]++
	return nil
}

func TestImagesPull(t *testing.T) {
	puller := &fakePuller{
		count: make(map[string]int),
	}
	images := []string{"a", "b", "c", "d", "a"}
	ip := cmd.NewImagesPull(puller, images)
	err := ip.PullAll()
	if err != nil {
		t.Fatalf("expected nil but found %v", err)
	}
	if puller.count["a"] != 2 {
		t.Fatalf("expected 2 but found %v", puller.count["a"])
	}
}

func TestMigrate(t *testing.T) {
	cfg := []byte(dedent.Dedent(`
		apiVersion: kubeadm.k8s.io/v1alpha2
		kind: MasterConfiguration
		kubernetesVersion: v1.10.0
	`))
	configFile, cleanup := tempConfig(t, cfg)
	defer cleanup()

	var output bytes.Buffer
	command := cmd.NewCmdConfigMigrate(&output)
	err := command.Flags().Set("old-config", configFile)
	if err != nil {
		t.Fatalf("failed to set old-config flag")
	}
	command.Run(nil, nil)
	_, err = config.BytesToInternalConfig(output.Bytes())
	if err != nil {
		t.Fatalf("Could not read output back into internal type: %v", err)
	}
}

// Returns the name of the file created and a cleanup callback
func tempConfig(t *testing.T, config []byte) (string, func()) {
	t.Helper()
	tmpDir, err := ioutil.TempDir("", "kubeadm-migration-test")
	if err != nil {
		t.Fatalf("Unable to create temporary directory: %v", err)
	}
	configFilePath := filepath.Join(tmpDir, "test-config-file")
	err = ioutil.WriteFile(configFilePath, config, 0644)
	if err != nil {
		os.RemoveAll(tmpDir)
		t.Fatalf("Failed writing a config file: %v", err)
	}
	return configFilePath, func() {
		os.RemoveAll(tmpDir)
	}
}
