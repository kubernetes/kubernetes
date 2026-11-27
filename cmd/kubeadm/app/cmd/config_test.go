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

	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeadmapiv1 "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/v1beta4"
	outputapischeme "k8s.io/kubernetes/cmd/kubeadm/app/apis/output/scheme"
	"k8s.io/kubernetes/cmd/kubeadm/app/constants"
	kubeadmutil "k8s.io/kubernetes/cmd/kubeadm/app/util"
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
kubernetesVersion: %s`, kubeadmapiv1.SchemeGroupVersion.String(), constants.CurrentKubernetesVersion))),
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
kubernetesVersion: %s`, kubeadmapiv1.SchemeGroupVersion.String(), constants.MinimumControlPlaneVersion))),
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
registry.k8s.io/coredns/coredns:{{.CoreDNSVersion}}
registry.k8s.io/pause:{{.PauseVersion}}
registry.k8s.io/etcd:{{.EtcdVersion}}
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
    "apiVersion": "output.kubeadm.k8s.io/v1alpha3",
    "images": [
        "registry.k8s.io/kube-apiserver:{{.KubeVersion}}",
        "registry.k8s.io/kube-controller-manager:{{.KubeVersion}}",
        "registry.k8s.io/kube-scheduler:{{.KubeVersion}}",
        "registry.k8s.io/kube-proxy:{{.KubeVersion}}",
        "registry.k8s.io/coredns/coredns:{{.CoreDNSVersion}}",
        "registry.k8s.io/pause:{{.PauseVersion}}",
        "registry.k8s.io/etcd:{{.EtcdVersion}}"
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
			expectedOutput: `apiVersion: output.kubeadm.k8s.io/v1alpha3
images:
- registry.k8s.io/kube-apiserver:{{.KubeVersion}}
- registry.k8s.io/kube-controller-manager:{{.KubeVersion}}
- registry.k8s.io/kube-scheduler:{{.KubeVersion}}
- registry.k8s.io/kube-proxy:{{.KubeVersion}}
- registry.k8s.io/coredns/coredns:{{.CoreDNSVersion}}
- registry.k8s.io/pause:{{.PauseVersion}}
- registry.k8s.io/etcd:{{.EtcdVersion}}
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
registry.k8s.io/coredns/coredns:{{.CoreDNSVersion}}
registry.k8s.io/pause:{{.PauseVersion}}
registry.k8s.io/etcd:{{.EtcdVersion}}
`,
		},
		{
			name: "JSONPATH output",
			cfg: kubeadmapiv1.ClusterConfiguration{
				KubernetesVersion: dummyKubernetesVersionStr,
			},
			outputFormat: `jsonpath={range.images[*]}{@} {end}`,
			expectedOutput: "registry.k8s.io/kube-apiserver:{{.KubeVersion}} registry.k8s.io/kube-controller-manager:{{.KubeVersion}} registry.k8s.io/kube-scheduler:{{.KubeVersion}} " +
				"registry.k8s.io/kube-proxy:{{.KubeVersion}} registry.k8s.io/coredns/coredns:{{.CoreDNSVersion}} registry.k8s.io/pause:{{.PauseVersion}} registry.k8s.io/etcd:{{.EtcdVersion}} ",
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
			name: "JoinConfiguration",
			expectedKinds: []string{
				constants.JoinConfigurationKind,
			},
			cmdProc: newCmdConfigPrintJoinDefaults,
		},
		{
			name:          "ResetConfiguration",
			expectedKinds: []string{constants.ResetConfigurationKind},
			cmdProc:       newCmdConfigPrintResetDefaults,
		},
		{
			name:          "UpgradeConfiguration",
			expectedKinds: []string{constants.UpgradeConfigurationKind},
			cmdProc:       newCmdConfigPrintUpgradeDefaults,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			var output bytes.Buffer

			command := test.cmdProc(&output)
			if test.componentConfigs != "" {
				if err := command.Flags().Set("component-configs", test.componentConfigs); err != nil {
					t.Fatalf("failed to set component-configs flag")
				}
			}
			if err := command.RunE(nil, nil); err != nil {
				t.Fatalf("Error from running the print command: %v", err)
			}

			gvkmap, err := kubeadmutil.SplitConfigDocuments(output.Bytes())
			if err != nil {
				t.Fatalf("unexpected failure of SplitConfigDocuments: %v", err)
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

func TestNewCmdConfigPrintActionDefaultsInvalidComponentConfigs(t *testing.T) {
	var out bytes.Buffer
	cmd := newCmdConfigPrintInitDefaults(&out)
	if err := cmd.Flags().Set("component-configs", "UnknownConfiguration"); err != nil {
		t.Fatalf("failed to set component-configs flag: %v", err)
	}

	err := cmd.RunE(nil, nil)
	if err == nil {
		t.Fatal("expected an error for an unsupported component config kind, got none")
	}
	if !strings.Contains(err.Error(), "--component-configs needs to contain some of") {
		t.Fatalf("unexpected error: %v", err)
	}
	if out.Len() != 0 {
		t.Errorf("expected no output on error, got %q", out.String())
	}
}

func TestNewCmdConfigSubcommands(t *testing.T) {
	tests := []struct {
		name     string
		cmdProc  func() *cobra.Command
		expected []string
	}{
		{
			name:     "config",
			cmdProc:  func() *cobra.Command { return newCmdConfig(io.Discard) },
			expected: []string{"images", "migrate", "print", "validate"},
		},
		{
			name:     "config print",
			cmdProc:  func() *cobra.Command { return newCmdConfigPrint(io.Discard) },
			expected: []string{"init-defaults", "join-defaults", "reset-defaults", "upgrade-defaults"},
		},
		{
			name:     "config images",
			cmdProc:  func() *cobra.Command { return newCmdConfigImages(io.Discard) },
			expected: []string{"list", "pull"},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			got := []string{}
			for _, subCmd := range test.cmdProc().Commands() {
				got = append(got, subCmd.Name())
			}
			sort.Strings(got)
			if !reflect.DeepEqual(got, test.expected) {
				t.Fatalf("subcommands not matching:\n\texpected: %v\n\tgot: %v\n", test.expected, got)
			}
		})
	}
}

func TestNewCmdConfigMigrate(t *testing.T) {
	oldConfigBytes, err := getDefaultNodeConfigBytes()
	if err != nil {
		t.Fatalf("failed to get the default join configuration: %v", err)
	}

	tmpDir := t.TempDir()
	oldCfgPath := filepath.Join(tmpDir, "old-config.yaml")
	if err := os.WriteFile(oldCfgPath, oldConfigBytes, 0644); err != nil {
		t.Fatalf("failed to write the old config file: %v", err)
	}
	invalidCfgPath := filepath.Join(tmpDir, "invalid-config.yaml")
	if err := os.WriteFile(invalidCfgPath, []byte(dedent.Dedent(`
		apiVersion: kubeadm.k8s.io/v1beta4
		kind: UnknownConfiguration`)), 0644); err != nil {
		t.Fatalf("failed to write the invalid config file: %v", err)
	}

	tests := []struct {
		name           string
		oldCfgPath     string
		writeNewConfig bool
		expectError    bool
	}{
		{
			name:        "--old-config is mandatory",
			expectError: true,
		},
		{
			name:        "old config file does not exist",
			oldCfgPath:  filepath.Join(tmpDir, "missing-config.yaml"),
			expectError: true,
		},
		{
			name:       "print the migrated config to stdout",
			oldCfgPath: oldCfgPath,
		},
		{
			name:           "write the migrated config to a file",
			oldCfgPath:     oldCfgPath,
			writeNewConfig: true,
		},
		{
			name:        "invalid old config",
			oldCfgPath:  invalidCfgPath,
			expectError: true,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			var out bytes.Buffer
			cmd := newCmdConfigMigrate(&out)

			if test.oldCfgPath != "" {
				if err := cmd.Flags().Set("old-config", test.oldCfgPath); err != nil {
					t.Fatalf("failed to set old-config flag: %v", err)
				}
			}
			newCfgPath := filepath.Join(t.TempDir(), "new-config.yaml")
			if test.writeNewConfig {
				if err := cmd.Flags().Set("new-config", newCfgPath); err != nil {
					t.Fatalf("failed to set new-config flag: %v", err)
				}
			}

			err := cmd.RunE(nil, nil)
			if test.expectError {
				if err == nil {
					t.Fatal("expected an error, got none")
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}

			migrated := out.String()
			if test.writeNewConfig {
				if out.Len() != 0 {
					t.Errorf("expected no output when writing to a file, got %q", migrated)
				}
				migratedBytes, err := os.ReadFile(newCfgPath)
				if err != nil {
					t.Fatalf("failed to read the new config file: %v", err)
				}
				migrated = string(migratedBytes)
			}
			if !strings.Contains(migrated, "kind: "+constants.JoinConfigurationKind) {
				t.Errorf("expected the migrated config to contain kind %s, got:\n%s", constants.JoinConfigurationKind, migrated)
			}
		})
	}
}

func TestNewCmdConfigValidate(t *testing.T) {
	validConfigBytes, err := getDefaultInitConfigBytes()
	if err != nil {
		t.Fatalf("failed to get the default init configuration: %v", err)
	}

	tmpDir := t.TempDir()
	validCfgPath := filepath.Join(tmpDir, "valid-config.yaml")
	if err := os.WriteFile(validCfgPath, validConfigBytes, 0644); err != nil {
		t.Fatalf("failed to write the valid config file: %v", err)
	}
	invalidCfgPath := filepath.Join(tmpDir, "invalid-config.yaml")
	if err := os.WriteFile(invalidCfgPath, []byte(dedent.Dedent(`
		apiVersion: kubeadm.k8s.io/v1beta4
		kind: ClusterConfiguration
		kubernetesVersion: not-a-version`)), 0644); err != nil {
		t.Fatalf("failed to write the invalid config file: %v", err)
	}

	tests := []struct {
		name        string
		cfgPath     string
		expectError bool
	}{
		{
			name:        "--config is mandatory",
			expectError: true,
		},
		{
			name:        "config file does not exist",
			cfgPath:     filepath.Join(tmpDir, "missing-config.yaml"),
			expectError: true,
		},
		{
			name:    "valid config",
			cfgPath: validCfgPath,
		},
		{
			name:        "invalid config",
			cfgPath:     invalidCfgPath,
			expectError: true,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			var out bytes.Buffer
			cmd := newCmdConfigValidate(&out)
			if test.cfgPath != "" {
				if err := cmd.Flags().Set("config", test.cfgPath); err != nil {
					t.Fatalf("failed to set config flag: %v", err)
				}
			}

			err := cmd.RunE(nil, nil)
			if test.expectError {
				if err == nil {
					t.Fatal("expected an error, got none")
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if out.String() != "ok\n" {
				t.Errorf("expected %q, got %q", "ok\n", out.String())
			}
		})
	}
}

func TestNewCmdConfigImagesPull(t *testing.T) {
	tests := []struct {
		name         string
		featureGates string
		cfgPath      string
		expectError  bool
	}{
		{
			name:         "invalid feature gate",
			featureGates: "UnknownFeatureGate=true",
			expectError:  true,
		},
		{
			name:        "config file does not exist",
			cfgPath:     filepath.Join(t.TempDir(), "missing-config.yaml"),
			expectError: true,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			cmd := newCmdConfigImagesPull()
			if test.featureGates != "" {
				if err := cmd.PersistentFlags().Set("feature-gates", test.featureGates); err != nil {
					t.Fatalf("failed to set feature-gates flag: %v", err)
				}
			}
			if test.cfgPath != "" {
				if err := cmd.PersistentFlags().Set("config", test.cfgPath); err != nil {
					t.Fatalf("failed to set config flag: %v", err)
				}
			}

			err := cmd.RunE(nil, nil)
			if test.expectError && err == nil {
				t.Fatal("expected an error, got none")
			}
			if !test.expectError && err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
		})
	}
}

func TestNewCmdConfigImagesList(t *testing.T) {
	tests := []struct {
		name         string
		featureGates string
		cfgPath      string
		outputFormat string
		expectError  bool
	}{
		{
			name:         "invalid feature gate",
			featureGates: "UnknownFeatureGate=true",
			expectError:  true,
		},
		{
			name:        "config file does not exist",
			cfgPath:     filepath.Join(t.TempDir(), "missing-config.yaml"),
			expectError: true,
		},
		{
			name:         "valid defaults with text output",
			outputFormat: "text",
		},
		{
			name:         "unsupported output format",
			outputFormat: "not-a-format",
			expectError:  true,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			var out bytes.Buffer
			mockVersion := constants.MinimumControlPlaneVersion.String()
			cmd := newCmdConfigImagesList(&out, &mockVersion)
			if test.featureGates != "" {
				if err := cmd.PersistentFlags().Set("feature-gates", test.featureGates); err != nil {
					t.Fatalf("failed to set feature-gates flag: %v", err)
				}
			}
			if test.cfgPath != "" {
				if err := cmd.PersistentFlags().Set("config", test.cfgPath); err != nil {
					t.Fatalf("failed to set config flag: %v", err)
				}
			}
			if test.outputFormat != "" {
				if err := cmd.Flags().Set("output", test.outputFormat); err != nil {
					t.Fatalf("failed to set output flag: %v", err)
				}
			}

			err := cmd.RunE(nil, nil)
			if test.expectError && err == nil {
				t.Fatal("expected an error, got none")
			}
			if !test.expectError && err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
		})
	}
}

func TestPullControlPlaneImages(t *testing.T) {
	tests := []struct {
		name        string
		pullErr     error
		expectError bool
	}{
		{
			name: "all images pulled",
		},
		{
			name:        "pull failure",
			pullErr:     fmt.Errorf("pull failed"),
			expectError: true,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			fakeRuntime := utilruntime.NewContainerRuntime("/var/run/containerd/containerd.sock")
			fakeImpl := &utilruntime.FakeImpl{}
			fakeImpl.PullImageReturns("", test.pullErr)
			fakeRuntime.SetImpl(fakeImpl)

			cfg := &kubeadmapi.ClusterConfiguration{
				KubernetesVersion: constants.MinimumControlPlaneVersion.String(),
			}
			err := PullControlPlaneImages(fakeRuntime, cfg)
			if test.expectError {
				if err == nil {
					t.Fatal("expected an error, got none")
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
		})
	}
}
