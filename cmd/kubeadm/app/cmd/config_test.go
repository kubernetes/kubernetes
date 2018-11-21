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
	"io"
	"io/ioutil"
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"strings"
	"testing"

	"github.com/renstrom/dedent"
	"github.com/spf13/cobra"
	kubeadmapiv1beta1 "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/v1beta1"
	"k8s.io/kubernetes/cmd/kubeadm/app/componentconfigs"
	"k8s.io/kubernetes/cmd/kubeadm/app/constants"
	kubeadmutil "k8s.io/kubernetes/cmd/kubeadm/app/util"
	configutil "k8s.io/kubernetes/cmd/kubeadm/app/util/config"
	utilruntime "k8s.io/kubernetes/cmd/kubeadm/app/util/runtime"
	"k8s.io/utils/exec"
	fakeexec "k8s.io/utils/exec/testing"
)

const (
	defaultNumberOfImages = 8
	// dummyKubernetesVersion is just used for unit testing, in order to not make
	// kubeadm lookup dl.k8s.io to resolve what the latest stable release is
	dummyKubernetesVersion = "v1.12.0"
)

func TestNewCmdConfigImagesList(t *testing.T) {
	var output bytes.Buffer
	mockK8sVersion := dummyKubernetesVersion
	images := NewCmdConfigImagesList(&output, &mockK8sVersion)
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
				":v1.12.1",
			},
			configContents: []byte(dedent.Dedent(`
				apiVersion: kubeadm.k8s.io/v1beta1
				kind: ClusterConfiguration
				kubernetesVersion: v1.12.1
			`)),
		},
		{
			name:               "use coredns",
			expectedImageCount: defaultNumberOfImages,
			expectedImageSubstrings: []string{
				"coredns",
			},
			configContents: []byte(dedent.Dedent(`
				apiVersion: kubeadm.k8s.io/v1beta1
				kind: ClusterConfiguration
				kubernetesVersion: v1.12.0
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
			if err := ioutil.WriteFile(configFilePath, tc.configContents, 0644); err != nil {
				t.Fatalf("Failed writing a config file: %v", err)
			}

			i, err := NewImagesList(configFilePath, &kubeadmapiv1beta1.InitConfiguration{
				ClusterConfiguration: kubeadmapiv1beta1.ClusterConfiguration{
					KubernetesVersion: dummyKubernetesVersion,
				},
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
		cfg            kubeadmapiv1beta1.InitConfiguration
		expectedImages int
	}{
		{
			name:           "empty config",
			expectedImages: defaultNumberOfImages,
			cfg: kubeadmapiv1beta1.InitConfiguration{
				ClusterConfiguration: kubeadmapiv1beta1.ClusterConfiguration{
					KubernetesVersion: dummyKubernetesVersion,
				},
			},
		},
		{
			name: "external etcd configuration",
			cfg: kubeadmapiv1beta1.InitConfiguration{
				ClusterConfiguration: kubeadmapiv1beta1.ClusterConfiguration{
					Etcd: kubeadmapiv1beta1.Etcd{
						External: &kubeadmapiv1beta1.ExternalEtcd{
							Endpoints: []string{"https://some.etcd.com:2379"},
						},
					},
					KubernetesVersion: dummyKubernetesVersion,
				},
			},
			expectedImages: defaultNumberOfImages - 1,
		},
		{
			name: "coredns enabled",
			cfg: kubeadmapiv1beta1.InitConfiguration{
				ClusterConfiguration: kubeadmapiv1beta1.ClusterConfiguration{
					KubernetesVersion: dummyKubernetesVersion,
				},
			},
			expectedImages: defaultNumberOfImages,
		},
		{
			name: "kube-dns enabled",
			cfg: kubeadmapiv1beta1.InitConfiguration{
				ClusterConfiguration: kubeadmapiv1beta1.ClusterConfiguration{
					KubernetesVersion: dummyKubernetesVersion,
					DNS: kubeadmapiv1beta1.DNS{
						Type: kubeadmapiv1beta1.KubeDNS,
					},
				},
			},
			expectedImages: defaultNumberOfImages + 2,
		},
	}

	for _, tc := range testcases {
		t.Run(tc.name, func(t *testing.T) {
			i, err := NewImagesList("", &tc.cfg)
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

func TestImagesPull(t *testing.T) {
	fcmd := fakeexec.FakeCmd{
		CombinedOutputScript: []fakeexec.FakeCombinedOutputAction{
			func() ([]byte, error) { return nil, nil },
			func() ([]byte, error) { return nil, nil },
			func() ([]byte, error) { return nil, nil },
			func() ([]byte, error) { return nil, nil },
			func() ([]byte, error) { return nil, nil },
		},
	}

	fexec := fakeexec.FakeExec{
		CommandScript: []fakeexec.FakeCommandAction{
			func(cmd string, args ...string) exec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
			func(cmd string, args ...string) exec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
			func(cmd string, args ...string) exec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
			func(cmd string, args ...string) exec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
			func(cmd string, args ...string) exec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
		},
		LookPathFunc: func(cmd string) (string, error) { return "/usr/bin/docker", nil },
	}

	containerRuntime, err := utilruntime.NewContainerRuntime(&fexec, kubeadmapiv1beta1.DefaultCRISocket)
	if err != nil {
		t.Errorf("unexpected NewContainerRuntime error: %v", err)
	}

	images := []string{"a", "b", "c", "d", "a"}
	ip := NewImagesPull(containerRuntime, images)

	err = ip.PullAll()
	if err != nil {
		t.Fatalf("expected nil but found %v", err)
	}

	if fcmd.CombinedOutputCalls != len(images) {
		t.Errorf("expected %d calls, got %d", len(images), fcmd.CombinedOutputCalls)
	}
}

func TestMigrate(t *testing.T) {
	cfg := []byte(dedent.Dedent(`
		# This is intentionally testing an old API version and the old kind naming and making sure the output is correct
		apiVersion: kubeadm.k8s.io/v1alpha3
		kind: InitConfiguration
	`))
	configFile, cleanup := tempConfig(t, cfg)
	defer cleanup()

	var output bytes.Buffer
	command := NewCmdConfigMigrate(&output)
	if err := command.Flags().Set("old-config", configFile); err != nil {
		t.Fatalf("failed to set old-config flag")
	}
	newConfigPath := filepath.Join(filepath.Dir(configFile), "new-migrated-config")
	if err := command.Flags().Set("new-config", newConfigPath); err != nil {
		t.Fatalf("failed to set new-config flag")
	}
	command.Run(nil, nil)
	if _, err := configutil.ConfigFileAndDefaultsToInternalConfig(newConfigPath, &kubeadmapiv1beta1.InitConfiguration{}); err != nil {
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
	if err := ioutil.WriteFile(configFilePath, config, 0644); err != nil {
		os.RemoveAll(tmpDir)
		t.Fatalf("Failed writing a config file: %v", err)
	}
	return configFilePath, func() {
		os.RemoveAll(tmpDir)
	}
}

func TestNewCmdConfigPrintActionDefaults(t *testing.T) {
	tests := []struct {
		name             string
		expectedKinds    []string // need to be sorted
		componentConfigs string
		cmdProc          func(out io.Writer) *cobra.Command
	}{
		{
			name: "InitConfiguration: No component configs",
			expectedKinds: []string{
				constants.ClusterConfigurationKind,
				constants.InitConfigurationKind,
			},
			cmdProc: NewCmdConfigPrintInitDefaults,
		},
		{
			name: "InitConfiguration: KubeProxyConfiguration",
			expectedKinds: []string{
				constants.ClusterConfigurationKind,
				constants.InitConfigurationKind,
				string(componentconfigs.KubeProxyConfigurationKind),
			},
			componentConfigs: "KubeProxyConfiguration",
			cmdProc:          NewCmdConfigPrintInitDefaults,
		},
		{
			name: "InitConfiguration: KubeProxyConfiguration and KubeletConfiguration",
			expectedKinds: []string{
				constants.ClusterConfigurationKind,
				constants.InitConfigurationKind,
				string(componentconfigs.KubeProxyConfigurationKind),
				string(componentconfigs.KubeletConfigurationKind),
			},
			componentConfigs: "KubeProxyConfiguration,KubeletConfiguration",
			cmdProc:          NewCmdConfigPrintInitDefaults,
		},
		{
			name: "JoinConfiguration: No component configs",
			expectedKinds: []string{
				constants.JoinConfigurationKind,
			},
			cmdProc: NewCmdConfigPrintJoinDefaults,
		},
		{
			name: "JoinConfiguration: KubeProxyConfiguration",
			expectedKinds: []string{
				constants.JoinConfigurationKind,
				string(componentconfigs.KubeProxyConfigurationKind),
			},
			componentConfigs: "KubeProxyConfiguration",
			cmdProc:          NewCmdConfigPrintJoinDefaults,
		},
		{
			name: "JoinConfiguration: KubeProxyConfiguration and KubeletConfiguration",
			expectedKinds: []string{
				constants.JoinConfigurationKind,
				string(componentconfigs.KubeProxyConfigurationKind),
				string(componentconfigs.KubeletConfigurationKind),
			},
			componentConfigs: "KubeProxyConfiguration,KubeletConfiguration",
			cmdProc:          NewCmdConfigPrintJoinDefaults,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			var output bytes.Buffer

			command := test.cmdProc(&output)
			if err := command.Flags().Set("component-configs", test.componentConfigs); err != nil {
				t.Fatalf("failed to set component-configs flag")
			}
			command.Run(nil, nil)

			gvkmap, err := kubeadmutil.SplitYAMLDocuments(output.Bytes())
			if err != nil {
				t.Fatalf("unexpected failure of SplitYAMLDocuments: %v", err)
			}

			gotKinds := []string{}
			for gvk := range gvkmap {
				gotKinds = append(gotKinds, gvk.Kind)
			}

			sort.Strings(gotKinds)

			if !reflect.DeepEqual(gotKinds, test.expectedKinds) {
				t.Fatalf("kinds not matching:\n\texpectedKinds: %v\n\tgotKinds: %v\n", test.expectedKinds, gotKinds)
			}
		})
	}
}
