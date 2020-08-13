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
	"io/ioutil"
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"strings"
	"testing"
	"text/template"

	"github.com/lithammer/dedent"
	"github.com/spf13/cobra"
	kubeadmapiv1beta2 "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/v1beta2"
	outputapischeme "k8s.io/kubernetes/cmd/kubeadm/app/apis/output/scheme"
	"k8s.io/kubernetes/cmd/kubeadm/app/constants"
	kubeadmutil "k8s.io/kubernetes/cmd/kubeadm/app/util"
	configutil "k8s.io/kubernetes/cmd/kubeadm/app/util/config"
	"k8s.io/kubernetes/cmd/kubeadm/app/util/output"
	utilruntime "k8s.io/kubernetes/cmd/kubeadm/app/util/runtime"
	"k8s.io/utils/exec"
	fakeexec "k8s.io/utils/exec/testing"
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
	images := NewCmdConfigImagesList(&output, &mockK8sVersion)
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
				apiVersion: kubeadm.k8s.io/v1beta2
				kind: ClusterConfiguration
				kubernetesVersion: %s
			`, constants.CurrentKubernetesVersion))),
		},
		{
			name:               "use coredns",
			expectedImageCount: defaultNumberOfImages,
			expectedImageSubstrings: []string{
				"coredns",
			},
			configContents: []byte(dedent.Dedent(fmt.Sprintf(`
				apiVersion: kubeadm.k8s.io/v1beta2
				kind: ClusterConfiguration
				kubernetesVersion: %s
			`, constants.MinimumControlPlaneVersion))),
		},
	}

	outputFlags := output.NewOutputFlags(&imageTextPrintFlags{}).WithTypeSetter(outputapischeme.Scheme).WithDefaultOutput(output.TextOutput)
	printer, err := outputFlags.ToPrinter()
	if err != nil {
		t.Fatalf("can't create printer for the output format %s: %+v", output.TextOutput, err)
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

			i, err := NewImagesList(configFilePath, &kubeadmapiv1beta2.ClusterConfiguration{
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
		cfg            kubeadmapiv1beta2.ClusterConfiguration
		expectedImages int
	}{
		{
			name:           "empty config",
			expectedImages: defaultNumberOfImages,
			cfg: kubeadmapiv1beta2.ClusterConfiguration{
				KubernetesVersion: dummyKubernetesVersionStr,
			},
		},
		{
			name: "external etcd configuration",
			cfg: kubeadmapiv1beta2.ClusterConfiguration{
				Etcd: kubeadmapiv1beta2.Etcd{
					External: &kubeadmapiv1beta2.ExternalEtcd{
						Endpoints: []string{"https://some.etcd.com:2379"},
					},
				},
				KubernetesVersion: dummyKubernetesVersionStr,
			},
			expectedImages: defaultNumberOfImages - 1,
		},
		{
			name: "coredns enabled",
			cfg: kubeadmapiv1beta2.ClusterConfiguration{
				KubernetesVersion: dummyKubernetesVersionStr,
			},
			expectedImages: defaultNumberOfImages,
		},
		{
			name: "kube-dns enabled",
			cfg: kubeadmapiv1beta2.ClusterConfiguration{
				KubernetesVersion: dummyKubernetesVersionStr,
				DNS: kubeadmapiv1beta2.DNS{
					Type: kubeadmapiv1beta2.KubeDNS,
				},
			},
			expectedImages: defaultNumberOfImages + 2,
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

	etcdVersion, ok := constants.SupportedEtcdVersion[uint8(dummyKubernetesVersion.Minor())]
	if !ok {
		t.Fatalf("cannot determine etcd version for Kubernetes version %s", dummyKubernetesVersionStr)
	}
	versionMapping := struct {
		EtcdVersion    string
		KubeVersion    string
		PauseVersion   string
		CoreDNSVersion string
	}{
		EtcdVersion:    etcdVersion,
		KubeVersion:    "v" + dummyKubernetesVersionStr,
		PauseVersion:   constants.PauseVersion,
		CoreDNSVersion: constants.CoreDNSVersion,
	}

	testcases := []struct {
		name           string
		cfg            kubeadmapiv1beta2.ClusterConfiguration
		outputFormat   string
		expectedOutput string
	}{
		{
			name: "text output",
			cfg: kubeadmapiv1beta2.ClusterConfiguration{
				KubernetesVersion: dummyKubernetesVersionStr,
			},
			outputFormat: "text",
			expectedOutput: `k8s.gcr.io/kube-apiserver:{{.KubeVersion}}
k8s.gcr.io/kube-controller-manager:{{.KubeVersion}}
k8s.gcr.io/kube-scheduler:{{.KubeVersion}}
k8s.gcr.io/kube-proxy:{{.KubeVersion}}
k8s.gcr.io/pause:{{.PauseVersion}}
k8s.gcr.io/etcd:{{.EtcdVersion}}
k8s.gcr.io/coredns:{{.CoreDNSVersion}}
`,
		},
		{
			name: "JSON output",
			cfg: kubeadmapiv1beta2.ClusterConfiguration{
				KubernetesVersion: dummyKubernetesVersionStr,
			},
			outputFormat: "json",
			expectedOutput: `{
    "kind": "Images",
    "apiVersion": "output.kubeadm.k8s.io/v1alpha1",
    "images": [
        "k8s.gcr.io/kube-apiserver:{{.KubeVersion}}",
        "k8s.gcr.io/kube-controller-manager:{{.KubeVersion}}",
        "k8s.gcr.io/kube-scheduler:{{.KubeVersion}}",
        "k8s.gcr.io/kube-proxy:{{.KubeVersion}}",
        "k8s.gcr.io/pause:{{.PauseVersion}}",
        "k8s.gcr.io/etcd:{{.EtcdVersion}}",
        "k8s.gcr.io/coredns:{{.CoreDNSVersion}}"
    ]
}
`,
		},
		{
			name: "YAML output",
			cfg: kubeadmapiv1beta2.ClusterConfiguration{
				KubernetesVersion: dummyKubernetesVersionStr,
			},
			outputFormat: "yaml",
			expectedOutput: `apiVersion: output.kubeadm.k8s.io/v1alpha1
images:
- k8s.gcr.io/kube-apiserver:{{.KubeVersion}}
- k8s.gcr.io/kube-controller-manager:{{.KubeVersion}}
- k8s.gcr.io/kube-scheduler:{{.KubeVersion}}
- k8s.gcr.io/kube-proxy:{{.KubeVersion}}
- k8s.gcr.io/pause:{{.PauseVersion}}
- k8s.gcr.io/etcd:{{.EtcdVersion}}
- k8s.gcr.io/coredns:{{.CoreDNSVersion}}
kind: Images
`,
		},
		{
			name: "go-template output",
			cfg: kubeadmapiv1beta2.ClusterConfiguration{
				KubernetesVersion: dummyKubernetesVersionStr,
			},
			outputFormat: `go-template={{range .images}}{{.}}{{"\n"}}{{end}}`,
			expectedOutput: `k8s.gcr.io/kube-apiserver:{{.KubeVersion}}
k8s.gcr.io/kube-controller-manager:{{.KubeVersion}}
k8s.gcr.io/kube-scheduler:{{.KubeVersion}}
k8s.gcr.io/kube-proxy:{{.KubeVersion}}
k8s.gcr.io/pause:{{.PauseVersion}}
k8s.gcr.io/etcd:{{.EtcdVersion}}
k8s.gcr.io/coredns:{{.CoreDNSVersion}}
`,
		},
		{
			name: "JSONPATH output",
			cfg: kubeadmapiv1beta2.ClusterConfiguration{
				KubernetesVersion: dummyKubernetesVersionStr,
			},
			outputFormat: `jsonpath={range.images[*]}{@} {end}`,
			expectedOutput: "k8s.gcr.io/kube-apiserver:{{.KubeVersion}} k8s.gcr.io/kube-controller-manager:{{.KubeVersion}} k8s.gcr.io/kube-scheduler:{{.KubeVersion}} " +
				"k8s.gcr.io/kube-proxy:{{.KubeVersion}} k8s.gcr.io/pause:{{.PauseVersion}} k8s.gcr.io/etcd:{{.EtcdVersion}} k8s.gcr.io/coredns:{{.CoreDNSVersion}} ",
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
		LookPathFunc: func(cmd string) (string, error) { return "/usr/bin/docker", nil },
	}

	containerRuntime, err := utilruntime.NewContainerRuntime(&fexec, constants.DefaultDockerCRISocket)
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
	cfg := []byte(dedent.Dedent(`
		# This is intentionally testing an old API version. Sometimes this may be the latest version (if no old configs are supported).
		apiVersion: kubeadm.k8s.io/v1beta1
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
				"KubeProxyConfiguration",
			},
			componentConfigs: "KubeProxyConfiguration",
			cmdProc:          NewCmdConfigPrintInitDefaults,
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
				"KubeProxyConfiguration",
			},
			componentConfigs: "KubeProxyConfiguration",
			cmdProc:          NewCmdConfigPrintJoinDefaults,
		},
		{
			name: "JoinConfiguration: KubeProxyConfiguration and KubeletConfiguration",
			expectedKinds: []string{
				constants.JoinConfigurationKind,
				"KubeProxyConfiguration",
				"KubeletConfiguration",
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
