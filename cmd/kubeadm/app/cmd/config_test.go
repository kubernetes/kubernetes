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
	"io"
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"strings"
	"testing"
	"text/template"

	"github.com/lithammer/dedent"
	"github.com/spf13/cobra"

	"k8s.io/utils/exec"
	fakeexec "k8s.io/utils/exec/testing"

	kubeadmapiv1old "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/v1beta2"
	kubeadmapiv1 "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/v1beta3"
	outputapischeme "k8s.io/kubernetes/cmd/kubeadm/app/apis/output/scheme"
	"k8s.io/kubernetes/cmd/kubeadm/app/constants"
	kubeadmutil "k8s.io/kubernetes/cmd/kubeadm/app/util"
	configutil "k8s.io/kubernetes/cmd/kubeadm/app/util/config"
	"k8s.io/kubernetes/cmd/kubeadm/app/util/output"
	utilruntime "k8s.io/kubernetes/cmd/kubeadm/app/util/runtime"
)

const (
	defaultNumberOfImages = 8
)

var (
	// dummyKubernetesVersion and dummyKubernetesVersionStr are just used for unit testing, in order to not make
	// kubeadm lookup dl.k8s.io to resolve what the latest stable release is
	dummyKubernetesVersion    = constants.MinimumControlPlaneVersion
	dummyKubernetesVersionStr = dummyKubernetesVersion.String()
)

func TestNewCmdConfigImagesList(t *testing.T) {
	var output bytes.Buffer
	mockK8sVersion := dummyKubernetesVersionStr
	images := newCmdConfigImagesList(&output, &mockK8sVersion)
	if err := images.RunE(nil, nil); err != nil {
		t.Fatalf("Error from running the images command: %v", err)
	}
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
				constants.CurrentKubernetesVersion.String(),
			},
			configContents: []byte(dedent.Dedent(fmt.Sprintf(`
				apiVersion: %s
				kind: ClusterConfiguration
				kubernetesVersion: %s
			`, kubeadmapiv1.SchemeGroupVersion.String(), constants.CurrentKubernetesVersion))),
		},
		{
			name:               "use coredns",
			expectedImageCount: defaultNumberOfImages,
			expectedImageSubstrings: []string{
				"coredns",
			},
			configContents: []byte(dedent.Dedent(fmt.Sprintf(`
				apiVersion: %s
				kind: ClusterConfiguration
				kubernetesVersion: %s
			`, kubeadmapiv1.SchemeGroupVersion.String(), constants.MinimumControlPlaneVersion))),
		},
	}

	outputFlags := output.NewOutputFlags(&imageTextPrintFlags{}).WithTypeSetter(outputapischeme.Scheme).WithDefaultOutput(output.TextOutput)
	printer, err := outputFlags.ToPrinter()
	if err != nil {
		t.Fatalf("can't create printer for the output format %s: %+v", output.TextOutput, err)
	}

	for _, tc := range testcases {
		t.Run(tc.name, func(t *testing.T) {
			tmpDir, err := os.MkdirTemp("", "kubeadm-images-test")
			if err != nil {
				t.Fatalf("Unable to create temporary directory: %v", err)
			}
			defer os.RemoveAll(tmpDir)

			configFilePath := filepath.Join(tmpDir, "test-config-file")
			if err := os.WriteFile(configFilePath, tc.configContents, 0644); err != nil {
				t.Fatalf("Failed writing a config file: %v", err)
			}

			i, err := NewImagesList(configFilePath, &kubeadmapiv1.ClusterConfiguration{
				KubernetesVersion: dummyKubernetesVersionStr,
			})
			if err != nil {
				t.Fatalf("Failed getting the kubeadm images command: %v", err)
			}
			var output bytes.Buffer
			if err = i.Run(&output, printer); err != nil {
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
		cfg            kubeadmapiv1.ClusterConfiguration
		expectedImages int
	}{
		{
			name:           "empty config",
			expectedImages: defaultNumberOfImages,
			cfg: kubeadmapiv1.ClusterConfiguration{
				KubernetesVersion: dummyKubernetesVersionStr,
			},
		},
		{
			name: "external etcd configuration",
			cfg: kubeadmapiv1.ClusterConfiguration{
				Etcd: kubeadmapiv1.Etcd{
					External: &kubeadmapiv1.ExternalEtcd{
						Endpoints: []string{"https://some.etcd.com:2379"},
					},
				},
				KubernetesVersion: dummyKubernetesVersionStr,
			},
			expectedImages: defaultNumberOfImages - 1,
		},
		{
			name: "coredns enabled",
			cfg: kubeadmapiv1.ClusterConfiguration{
				KubernetesVersion: dummyKubernetesVersionStr,
			},
			expectedImages: defaultNumberOfImages,
		},
	}

	outputFlags := output.NewOutputFlags(&imageTextPrintFlags{}).WithTypeSetter(outputapischeme.Scheme).WithDefaultOutput(output.TextOutput)
	printer, err := outputFlags.ToPrinter()
	if err != nil {
		t.Fatalf("can't create printer for the output format %s: %+v", output.TextOutput, err)
	}
	for _, tc := range testcases {
		t.Run(tc.name, func(t *testing.T) {
			i, err := NewImagesList("", &tc.cfg)
			if err != nil {
				t.Fatalf("did not expect an error while creating the Images command: %v", err)
			}

			var output bytes.Buffer

			if err = i.Run(&output, printer); err != nil {
				t.Fatalf("did not expect an error running the Images command: %v", err)
			}

			actual := strings.Split(output.String(), "\n")
			if len(actual) != tc.expectedImages {
				t.Fatalf("expected %v images but got %v", tc.expectedImages, actual)
			}
		})
	}
}

func TestConfigImagesListOutput(t *testing.T) {

	etcdVersion, _, err := constants.EtcdSupportedVersion(constants.SupportedEtcdVersion, dummyKubernetesVersionStr)
	if err != nil {
		t.Fatalf("cannot determine etcd version for Kubernetes version %s", dummyKubernetesVersionStr)
	}
	versionMapping := struct {
		EtcdVersion    string
		KubeVersion    string
		PauseVersion   string
		CoreDNSVersion string
	}{
		EtcdVersion:    etcdVersion.String(),
		KubeVersion:    "v" + dummyKubernetesVersionStr,
		PauseVersion:   constants.PauseVersion,
		CoreDNSVersion: constants.CoreDNSVersion,
	}

	testcases := []struct {
		name           string
		cfg            kubeadmapiv1.ClusterConfiguration
		outputFormat   string
		expectedOutput string
	}{
		{
			name: "text output",
			cfg: kubeadmapiv1.ClusterConfiguration{
				KubernetesVersion: dummyKubernetesVersionStr,
			},
			outputFormat: "text",
			expectedOutput: `registry.k8s.io/kube-apiserver:{{.KubeVersion}}
registry.k8s.io/kube-controller-manager:{{.KubeVersion}}
registry.k8s.io/kube-scheduler:{{.KubeVersion}}
registry.k8s.io/kube-proxy:{{.KubeVersion}}
registry.k8s.io/pause:{{.PauseVersion}}
registry.k8s.io/etcd:{{.EtcdVersion}}
registry.k8s.io/coredns/coredns:{{.CoreDNSVersion}}
`,
		},
		{
			name: "JSON output",
			cfg: kubeadmapiv1.ClusterConfiguration{
				KubernetesVersion: dummyKubernetesVersionStr,
			},
			outputFormat: "json",
			expectedOutput: `{
    "kind": "Images",
    "apiVersion": "output.kubeadm.k8s.io/v1alpha2",
    "images": [
        "registry.k8s.io/kube-apiserver:{{.KubeVersion}}",
        "registry.k8s.io/kube-controller-manager:{{.KubeVersion}}",
        "registry.k8s.io/kube-scheduler:{{.KubeVersion}}",
        "registry.k8s.io/kube-proxy:{{.KubeVersion}}",
        "registry.k8s.io/pause:{{.PauseVersion}}",
        "registry.k8s.io/etcd:{{.EtcdVersion}}",
        "registry.k8s.io/coredns/coredns:{{.CoreDNSVersion}}"
    ]
}
`,
		},
		{
			name: "YAML output",
			cfg: kubeadmapiv1.ClusterConfiguration{
				KubernetesVersion: dummyKubernetesVersionStr,
			},
			outputFormat: "yaml",
			expectedOutput: `apiVersion: output.kubeadm.k8s.io/v1alpha2
images:
- registry.k8s.io/kube-apiserver:{{.KubeVersion}}
- registry.k8s.io/kube-controller-manager:{{.KubeVersion}}
- registry.k8s.io/kube-scheduler:{{.KubeVersion}}
- registry.k8s.io/kube-proxy:{{.KubeVersion}}
- registry.k8s.io/pause:{{.PauseVersion}}
- registry.k8s.io/etcd:{{.EtcdVersion}}
- registry.k8s.io/coredns/coredns:{{.CoreDNSVersion}}
kind: Images
`,
		},
		{
			name: "go-template output",
			cfg: kubeadmapiv1.ClusterConfiguration{
				KubernetesVersion: dummyKubernetesVersionStr,
			},
			outputFormat: `go-template={{range .images}}{{.}}{{"\n"}}{{end}}`,
			expectedOutput: `registry.k8s.io/kube-apiserver:{{.KubeVersion}}
registry.k8s.io/kube-controller-manager:{{.KubeVersion}}
registry.k8s.io/kube-scheduler:{{.KubeVersion}}
registry.k8s.io/kube-proxy:{{.KubeVersion}}
registry.k8s.io/pause:{{.PauseVersion}}
registry.k8s.io/etcd:{{.EtcdVersion}}
registry.k8s.io/coredns/coredns:{{.CoreDNSVersion}}
`,
		},
		{
			name: "JSONPATH output",
			cfg: kubeadmapiv1.ClusterConfiguration{
				KubernetesVersion: dummyKubernetesVersionStr,
			},
			outputFormat: `jsonpath={range.images[*]}{@} {end}`,
			expectedOutput: "registry.k8s.io/kube-apiserver:{{.KubeVersion}} registry.k8s.io/kube-controller-manager:{{.KubeVersion}} registry.k8s.io/kube-scheduler:{{.KubeVersion}} " +
				"registry.k8s.io/kube-proxy:{{.KubeVersion}} registry.k8s.io/pause:{{.PauseVersion}} registry.k8s.io/etcd:{{.EtcdVersion}} registry.k8s.io/coredns/coredns:{{.CoreDNSVersion}} ",
		},
	}

	for _, tc := range testcases {
		outputFlags := output.NewOutputFlags(&imageTextPrintFlags{}).WithTypeSetter(outputapischeme.Scheme).WithDefaultOutput(tc.outputFormat)
		printer, err := outputFlags.ToPrinter()
		if err != nil {
			t.Fatalf("can't create printer for the output format %s: %+v", tc.outputFormat, err)
		}

		t.Run(tc.name, func(t *testing.T) {
			i, err := NewImagesList("", &tc.cfg)
			if err != nil {
				t.Fatalf("did not expect an error while creating the Images command: %v", err)
			}

			var output, expectedOutput bytes.Buffer

			if err = i.Run(&output, printer); err != nil {
				t.Fatalf("did not expect an error running the Images command: %v", err)
			}

			tmpl, err := template.New("test").Parse(tc.expectedOutput)
			if err != nil {
				t.Fatalf("could not create template: %v", err)
			}
			if err = tmpl.Execute(&expectedOutput, versionMapping); err != nil {
				t.Fatalf("could not execute template: %v", err)
			}

			if output.String() != expectedOutput.String() {
				t.Fatalf("unexpected output:\n|%s|\nexpected:\n|%s|\n", output.String(), tc.expectedOutput)
			}
		})
	}
}

func TestImagesPull(t *testing.T) {
	fcmd := fakeexec.FakeCmd{
		CombinedOutputScript: []fakeexec.FakeAction{
			func() ([]byte, []byte, error) { return nil, nil, nil },
			func() ([]byte, []byte, error) { return nil, nil, nil },
			func() ([]byte, []byte, error) { return nil, nil, nil },
			func() ([]byte, []byte, error) { return nil, nil, nil },
			func() ([]byte, []byte, error) { return nil, nil, nil },
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
		LookPathFunc: func(cmd string) (string, error) { return "/usr/bin/crictl", nil },
	}

	containerRuntime, err := utilruntime.NewContainerRuntime(&fexec, constants.DefaultCRISocket)
	if err != nil {
		t.Errorf("unexpected NewContainerRuntime error: %v", err)
	}

	images := []string{"a", "b", "c", "d", "a"}
	for _, image := range images {
		if err := containerRuntime.PullImage(image); err != nil {
			t.Fatalf("expected nil but found %v", err)
		}
		fmt.Printf("[config/images] Pulled %s\n", image)
	}

	if fcmd.CombinedOutputCalls != len(images) {
		t.Errorf("expected %d calls, got %d", len(images), fcmd.CombinedOutputCalls)
	}
}

func TestMigrate(t *testing.T) {
	cfg := []byte(dedent.Dedent(fmt.Sprintf(`
		# This is intentionally testing an old API version. Sometimes this may be the latest version (if no old configs are supported).
		apiVersion: %s
		kind: InitConfiguration
	`, kubeadmapiv1old.SchemeGroupVersion.String())))
	configFile, cleanup := tempConfig(t, cfg)
	defer cleanup()

	var output bytes.Buffer
	command := newCmdConfigMigrate(&output)
	if err := command.Flags().Set("old-config", configFile); err != nil {
		t.Fatalf("failed to set old-config flag")
	}
	newConfigPath := filepath.Join(filepath.Dir(configFile), "new-migrated-config")
	if err := command.Flags().Set("new-config", newConfigPath); err != nil {
		t.Fatalf("failed to set new-config flag")
	}
	if err := command.RunE(nil, nil); err != nil {
		t.Fatalf("Error from running the migrate command: %v", err)
	}
	if _, err := configutil.LoadInitConfigurationFromFile(newConfigPath); err != nil {
		t.Fatalf("Could not read output back into internal type: %v", err)
	}
}

// Returns the name of the file created and a cleanup callback
func tempConfig(t *testing.T, config []byte) (string, func()) {
	t.Helper()
	tmpDir, err := os.MkdirTemp("", "kubeadm-migration-test")
	if err != nil {
		t.Fatalf("Unable to create temporary directory: %v", err)
	}
	configFilePath := filepath.Join(tmpDir, "test-config-file")
	if err := os.WriteFile(configFilePath, config, 0644); err != nil {
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
			cmdProc: newCmdConfigPrintInitDefaults,
		},
		{
			name: "InitConfiguration: KubeProxyConfiguration",
			expectedKinds: []string{
				constants.ClusterConfigurationKind,
				constants.InitConfigurationKind,
				"KubeProxyConfiguration",
			},
			componentConfigs: "KubeProxyConfiguration",
			cmdProc:          newCmdConfigPrintInitDefaults,
		},
		{
			name: "InitConfiguration: KubeProxyConfiguration and KubeletConfiguration",
			expectedKinds: []string{
				constants.ClusterConfigurationKind,
				constants.InitConfigurationKind,
				"KubeProxyConfiguration",
				"KubeletConfiguration",
			},
			componentConfigs: "KubeProxyConfiguration,KubeletConfiguration",
			cmdProc:          newCmdConfigPrintInitDefaults,
		},
		{
			name: "JoinConfiguration: No component configs",
			expectedKinds: []string{
				constants.JoinConfigurationKind,
			},
			cmdProc: newCmdConfigPrintJoinDefaults,
		},
		{
			name: "JoinConfiguration: KubeProxyConfiguration",
			expectedKinds: []string{
				constants.JoinConfigurationKind,
				"KubeProxyConfiguration",
			},
			componentConfigs: "KubeProxyConfiguration",
			cmdProc:          newCmdConfigPrintJoinDefaults,
		},
		{
			name: "JoinConfiguration: KubeProxyConfiguration and KubeletConfiguration",
			expectedKinds: []string{
				constants.JoinConfigurationKind,
				"KubeProxyConfiguration",
				"KubeletConfiguration",
			},
			componentConfigs: "KubeProxyConfiguration,KubeletConfiguration",
			cmdProc:          newCmdConfigPrintJoinDefaults,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			var output bytes.Buffer

			command := test.cmdProc(&output)
			if err := command.Flags().Set("component-configs", test.componentConfigs); err != nil {
				t.Fatalf("failed to set component-configs flag")
			}
			if err := command.RunE(nil, nil); err != nil {
				t.Fatalf("Error from running the print command: %v", err)
			}

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
