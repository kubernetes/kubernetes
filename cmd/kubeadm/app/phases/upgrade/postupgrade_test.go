/*
Copyright 2017 The Kubernetes Authors.

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
	"io"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/diff"
	errorsutil "k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/client-go/kubernetes/fake"
	k8stesting "k8s.io/client-go/testing"

	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	"k8s.io/kubernetes/cmd/kubeadm/app/componentconfigs"
	"k8s.io/kubernetes/cmd/kubeadm/app/constants"
	kubeletphase "k8s.io/kubernetes/cmd/kubeadm/app/phases/kubelet"
	"k8s.io/kubernetes/cmd/kubeadm/app/util/errors"
)

func TestMoveFiles(t *testing.T) {
	tmpdir := t.TempDir()

	certPath := filepath.Join(tmpdir, constants.APIServerCertName)
	certFile, err := os.OpenFile(certPath, os.O_CREATE|os.O_TRUNC|os.O_WRONLY, 0666)
	if err != nil {
		t.Fatalf("Failed to create cert file %s: %v", certPath, err)
	}
	certFile.Close()

	keyPath := filepath.Join(tmpdir, constants.APIServerKeyName)
	keyFile, err := os.OpenFile(keyPath, os.O_CREATE|os.O_TRUNC|os.O_WRONLY, 0666)
	if err != nil {
		t.Fatalf("Failed to create key file %s: %v", keyPath, err)
	}
	keyFile.Close()

	subDir := filepath.Join(tmpdir, "expired")
	if err := os.Mkdir(subDir, 0766); err != nil {
		t.Fatalf("Failed to create backup directory %s: %v", subDir, err)
	}

	filesToMove := map[string]string{
		filepath.Join(tmpdir, constants.APIServerCertName): filepath.Join(subDir, constants.APIServerCertName),
		filepath.Join(tmpdir, constants.APIServerKeyName):  filepath.Join(subDir, constants.APIServerKeyName),
	}

	if err := moveFiles(filesToMove); err != nil {
		t.Fatalf("Failed to move files %v: %v", filesToMove, err)
	}
}

func TestRollbackFiles(t *testing.T) {
	tmpdir := t.TempDir()

	subDir := filepath.Join(tmpdir, "expired")
	if err := os.Mkdir(subDir, 0766); err != nil {
		t.Fatalf("Failed to create backup directory %s: %v", subDir, err)
	}

	certPath := filepath.Join(subDir, constants.APIServerCertName)
	certFile, err := os.OpenFile(certPath, os.O_CREATE|os.O_TRUNC|os.O_WRONLY, 0666)
	if err != nil {
		t.Fatalf("Failed to create cert file %s: %v", certPath, err)
	}
	defer certFile.Close()

	keyPath := filepath.Join(subDir, constants.APIServerKeyName)
	keyFile, err := os.OpenFile(keyPath, os.O_CREATE|os.O_TRUNC|os.O_WRONLY, 0666)
	if err != nil {
		t.Fatalf("Failed to create key file %s: %v", keyPath, err)
	}
	defer keyFile.Close()

	filesToRollBack := map[string]string{
		filepath.Join(subDir, constants.APIServerCertName): filepath.Join(tmpdir, constants.APIServerCertName),
		filepath.Join(subDir, constants.APIServerKeyName):  filepath.Join(tmpdir, constants.APIServerKeyName),
	}

	errString := "there are files need roll back"
	originalErr := errors.New(errString)
	err = rollbackFiles(filesToRollBack, originalErr)
	if err == nil {
		t.Fatalf("Expected error contains %q, got nil", errString)
	}
	if !strings.Contains(err.Error(), errString) {
		t.Fatalf("Expected error contains %q, got %v", errString, err)
	}
}

func TestWriteKubeletConfigFiles(t *testing.T) {
	const existingInstanceConfig = `apiVersion: kubelet.config.k8s.io/v1beta1
kind: KubeletConfiguration
containerRuntimeEndpoint: unix:///var/run/existing.sock
`

	instanceConfigPath := func(kubeletDir string) string {
		return filepath.Join(kubeletDir, constants.KubeletInstanceConfigurationFileName)
	}

	testCases := []struct {
		name              string
		patchesDir        string
		dryRun            bool
		kubeletDirNotADir bool
		expectedErrMsg    string
		expectedError     bool
		cfg               *kubeadmapi.InitConfiguration
		setup             func(t *testing.T, kubeletDir string)
		verify            func(t *testing.T, kubeletDir string)
	}{
		{
			name:   "write kubelet config file successfully",
			dryRun: true,
			cfg: &kubeadmapi.InitConfiguration{
				ClusterConfiguration: kubeadmapi.ClusterConfiguration{
					ComponentConfigs: kubeadmapi.ComponentConfigMap{
						componentconfigs.KubeletGroup: &componentConfig{},
					},
				},
			},
		},
		{
			name:          "aggregate errs: no kubelet config file and cannot read config file",
			dryRun:        true,
			expectedError: true,
			cfg:           &kubeadmapi.InitConfiguration{},
		},
		{
			name:          "only one err: patch dir does not exist",
			patchesDir:    "Bogus",
			dryRun:        true,
			expectedError: true,
			cfg: &kubeadmapi.InitConfiguration{
				ClusterConfiguration: kubeadmapi.ClusterConfiguration{
					ComponentConfigs: kubeadmapi.ComponentConfigMap{
						componentconfigs.KubeletGroup: &componentConfig{},
					},
				},
			},
		},
		{
			name:   "missing instance config file is created with defaults",
			dryRun: true,
			cfg: &kubeadmapi.InitConfiguration{
				ClusterConfiguration: kubeadmapi.ClusterConfiguration{
					ComponentConfigs: kubeadmapi.ComponentConfigMap{
						componentconfigs.KubeletGroup: &componentConfig{},
					},
				},
			},
			verify: func(t *testing.T, kubeletDir string) {
				if _, err := os.Stat(instanceConfigPath(kubeletDir)); err != nil {
					t.Fatalf("expected %s to be created, got: %v",
						constants.KubeletInstanceConfigurationFileName, err)
				}
			},
		},
		{
			name:   "existing instance config file is left alone",
			dryRun: true,
			cfg: &kubeadmapi.InitConfiguration{
				ClusterConfiguration: kubeadmapi.ClusterConfiguration{
					ComponentConfigs: kubeadmapi.ComponentConfigMap{
						componentconfigs.KubeletGroup: &componentConfig{},
					},
				},
			},
			setup: func(t *testing.T, kubeletDir string) {
				if err := os.WriteFile(instanceConfigPath(kubeletDir), []byte(existingInstanceConfig), 0644); err != nil {
					t.Fatalf("failed to write %s: %v",
						constants.KubeletInstanceConfigurationFileName, err)
				}
			},
			verify: func(t *testing.T, kubeletDir string) {
				got, err := os.ReadFile(instanceConfigPath(kubeletDir))
				if err != nil {
					t.Fatalf("failed to read %s: %v",
						constants.KubeletInstanceConfigurationFileName, err)
				}
				if string(got) != existingInstanceConfig {
					t.Fatalf("expected %s to be left alone, got:\n%s",
						constants.KubeletInstanceConfigurationFileName, got)
				}
			},
		},
		{
			name: "kubelet config file is backed up before being rewritten",
			cfg: &kubeadmapi.InitConfiguration{
				ClusterConfiguration: kubeadmapi.ClusterConfiguration{
					ComponentConfigs: kubeadmapi.ComponentConfigMap{
						componentconfigs.KubeletGroup: &componentConfig{},
					},
				},
			},
			setup: func(t *testing.T, kubeletDir string) {
				if err := os.WriteFile(filepath.Join(kubeletDir, constants.KubeletConfigurationFileName), []byte(existingInstanceConfig), 0644); err != nil {
					t.Fatalf("failed to write %s: %v", constants.KubeletConfigurationFileName, err)
				}
			},
			verify: func(t *testing.T, kubeletDir string) {
				backups, err := filepath.Glob(filepath.Join(kubeletDir, constants.TempDir, "kubeadm-kubelet-config-*", constants.KubeletConfigurationFileName))
				if err != nil {
					t.Fatalf("failed to look up the kubelet config backup: %v", err)
				}
				if len(backups) != 1 {
					t.Fatalf("expected exactly one kubelet config backup, got %v", backups)
				}
				got, err := os.ReadFile(backups[0])
				if err != nil {
					t.Fatalf("failed to read the kubelet config backup: %v", err)
				}
				if string(got) != existingInstanceConfig {
					t.Fatalf("expected the backup to contain the previous config, got:\n%s", got)
				}
			},
		},
		{
			name:          "backing up a missing kubelet config file fails",
			expectedError: true,
			cfg: &kubeadmapi.InitConfiguration{
				ClusterConfiguration: kubeadmapi.ClusterConfiguration{
					ComponentConfigs: kubeadmapi.ComponentConfigMap{
						componentconfigs.KubeletGroup: &componentConfig{},
					},
				},
			},
		},
		{
			name:              "stating the instance config file fails with an error other than not-exist",
			kubeletDirNotADir: true,
			dryRun:            true,
			expectedError:     true,
			expectedErrMsg:    "stat error for the kubelet instance configuration file",
			cfg: &kubeadmapi.InitConfiguration{
				ClusterConfiguration: kubeadmapi.ClusterConfiguration{
					ComponentConfigs: kubeadmapi.ComponentConfigMap{
						componentconfigs.KubeletGroup: &componentConfig{},
					},
				},
			},
		},
	}
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			kubeletDir := t.TempDir()
			kubeConfigDir := kubeletDir
			if tc.kubeletDirNotADir {
				notADir := filepath.Join(t.TempDir(), "not-a-dir")
				if err := os.WriteFile(notADir, []byte("not a directory"), 0644); err != nil {
					t.Fatalf("failed to create %s: %v", notADir, err)
				}
				kubeletDir = filepath.Join(notADir, "kubelet")
			}
			if tc.setup != nil {
				tc.setup(t, kubeletDir)
			}
			err := WriteKubeletConfigFiles(tc.cfg, kubeletDir, kubeConfigDir, tc.patchesDir, tc.dryRun, io.Discard)
			if (err != nil) != tc.expectedError {
				t.Fatalf("expected error: %v, got: %v, error: %v", tc.expectedError, err != nil, err)
			}
			if tc.expectedErrMsg != "" && !strings.Contains(err.Error(), tc.expectedErrMsg) {
				t.Fatalf("expected error containing %q, got: %v", tc.expectedErrMsg, err)
			}
			if tc.verify != nil {
				tc.verify(t, kubeletDir)
			}
		})
	}
}

func TestRemoveKubeletArgsFromFile(t *testing.T) {
	testCases := []struct {
		name            string
		kubeletFlags    []kubeadmapi.Arg
		unwantedFlags   []string
		dryRun          bool
		skipWrite       bool
		wantErr         bool
		wantUnchanged   bool
		wantFileContent string
	}{
		{
			name: "no unwanted flags leaves the file untouched",
			kubeletFlags: []kubeadmapi.Arg{
				{Name: "node-ip", Value: "172.18.0.2"},
			},
			wantUnchanged: true,
		},
		{
			name: "dry run leaves the file untouched",
			kubeletFlags: []kubeadmapi.Arg{
				{Name: "node-ip", Value: "172.18.0.2"},
				{Name: "pod-infra-container-image", Value: "registry.k8s.io/pause:ver"},
			},
			unwantedFlags: []string{"pod-infra-container-image"},
			dryRun:        true,
			wantUnchanged: true,
		},
		{
			name:          "missing kubelet env file",
			unwantedFlags: []string{"pod-infra-container-image"},
			skipWrite:     true,
			wantErr:       true,
		},
		{
			name: "remove an existing flag",
			kubeletFlags: []kubeadmapi.Arg{
				{Name: "node-ip", Value: "172.18.0.2"},
				{Name: "node-labels", Value: ""},
				{Name: "pod-infra-container-image", Value: "registry.k8s.io/pause:ver"},
				{Name: "provider-id", Value: "kind://docker/kind/kind-control-plane"},
			},
			unwantedFlags: []string{
				"pod-infra-container-image",
			},
			wantErr: false,
			wantFileContent: `KUBELET_KUBEADM_ARGS="--node-ip=172.18.0.2 --node-labels= --provider-id=kind://docker/kind/kind-control-plane"
`,
		},
		{
			name: "remove multiple existing flags",
			kubeletFlags: []kubeadmapi.Arg{
				{Name: "node-ip", Value: "172.18.0.2"},
				{Name: "node-labels", Value: ""},
				{Name: "pod-infra-container-image", Value: "registry.k8s.io/pause:ver"},
				{Name: "provider-id", Value: "kind://docker/kind/kind-control-plane"},
			},
			unwantedFlags: []string{
				"pod-infra-container-image",
				"node-labels",
			},
			wantErr: false,
			wantFileContent: `KUBELET_KUBEADM_ARGS="--node-ip=172.18.0.2 --provider-id=kind://docker/kind/kind-control-plane"
`,
		},
		{
			name: "remove non-existing flags",
			kubeletFlags: []kubeadmapi.Arg{
				{Name: "node-ip", Value: "172.18.0.2"},
				{Name: "node-labels", Value: ""},
				{Name: "pod-infra-container-image", Value: "registry.k8s.io/pause:ver"},
				{Name: "provider-id", Value: "kind://docker/kind/kind-control-plane"},
			},
			unwantedFlags: []string{
				"foo",
			},
			wantErr: false,
			wantFileContent: `KUBELET_KUBEADM_ARGS="--node-ip=172.18.0.2 --node-labels= --pod-infra-container-image=registry.k8s.io/pause:ver --provider-id=kind://docker/kind/kind-control-plane"
`,
		},
		{
			name: "remove multiple flags mixed with non-existing and existing flags",
			kubeletFlags: []kubeadmapi.Arg{
				{Name: "node-ip", Value: "172.18.0.2"},
				{Name: "node-labels", Value: ""},
				{Name: "pod-infra-container-image", Value: "registry.k8s.io/pause:ver"},
				{Name: "provider-id", Value: "kind://docker/kind/kind-control-plane"},
			},
			unwantedFlags: []string{
				"pod-infra-container-image",
				"foo",
			},
			wantErr: false,
			wantFileContent: `KUBELET_KUBEADM_ARGS="--node-ip=172.18.0.2 --node-labels= --provider-id=kind://docker/kind/kind-control-plane"
`,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			tempDir := t.TempDir()

			before := ""
			if !tc.skipWrite {
				err := kubeletphase.WriteKubeletArgsToFile(tc.kubeletFlags, nil, tempDir)
				if err != nil {
					t.Fatalf("Failed to write kubeadm-flags.env file: %v", err)
				}
				data, err := os.ReadFile(filepath.Join(tempDir, constants.KubeletEnvFileName))
				if err != nil {
					t.Fatalf("Failed to read kubelet.env file: %v", err)
				}
				before = string(data)
			}

			err := RemoveKubeletArgsFromFile(tempDir, tempDir, tc.unwantedFlags, tc.dryRun, io.Discard)
			if (err != nil) != tc.wantErr {
				t.Fatalf("expected error: %v, got: %v, error: %v", tc.wantErr, err != nil, err)
			}
			if tc.wantErr {
				return
			}

			kubeletEnvFilePath := filepath.Join(tempDir, constants.KubeletEnvFileName)
			fileContent, err := os.ReadFile(kubeletEnvFilePath)
			if err != nil {
				t.Fatalf("Failed to read kubelet.env file: %v", err)
			}
			want := tc.wantFileContent
			if tc.wantUnchanged {
				want = before
			}
			if gotOut := string(fileContent); gotOut != want {
				t.Fatalf("Actual modified content of RemoveKubeletArgsFromFile() does not match expected.\nActual:  %v\nExpected: %v\n, Diff: %v", gotOut, want, diff.Diff(gotOut, want))
			}
		})
	}
}

func TestUnupgradedControlPlaneInstances(t *testing.T) {
	testCases := []struct {
		name          string
		pods          []corev1.Pod
		currentNode   string
		listErr       error
		expectedNodes []string
		expectError   bool
	}{
		{
			name:        "cannot list kube-apiserver Pods",
			pods:        []corev1.Pod{},
			currentNode: "node-1",
			listErr:     errors.New("list failed"),
			expectError: true,
		},
		{
			name: "Pod without a kube-apiserver container",
			pods: []corev1.Pod{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "kube-apiserver-1",
						Namespace: metav1.NamespaceSystem,
						Labels: map[string]string{
							"component": constants.KubeAPIServer,
						},
					},
					Spec: corev1.PodSpec{
						NodeName: "node-1",
						Containers: []corev1.Container{
							{Name: "not-kube-apiserver", Image: "registry.kl8s.io/kube-apiserver:v2"},
						},
					},
				},
			},
			currentNode: "node-1",
			expectError: true,
		},
		{
			name: "Pod for the current control plane instance is missing",
			pods: []corev1.Pod{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "kube-apiserver-1",
						Namespace: metav1.NamespaceSystem,
						Labels: map[string]string{
							"component": constants.KubeAPIServer,
						},
					},
					Spec: corev1.PodSpec{
						NodeName: "node-1",
						Containers: []corev1.Container{
							{Name: constants.KubeAPIServer, Image: "registry.kl8s.io/kube-apiserver:v2"},
						},
					},
				},
			},
			currentNode: "node-2",
			expectError: true,
		},
		{
			name: "two nodes, one needs upgrade",
			pods: []corev1.Pod{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "kube-apiserver-1",
						Namespace: metav1.NamespaceSystem,
						Labels: map[string]string{
							"component": constants.KubeAPIServer,
						},
					},
					Spec: corev1.PodSpec{
						NodeName: "node-1",
						Containers: []corev1.Container{
							{Name: constants.KubeAPIServer, Image: "registry.kl8s.io/kube-apiserver:v2"},
						},
					},
				},
				{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "kube-apiserver-2",
						Namespace: metav1.NamespaceSystem,
						Labels: map[string]string{
							"component": constants.KubeAPIServer,
						},
					},
					Spec: corev1.PodSpec{
						NodeName: "node-2",
						Containers: []corev1.Container{
							{Name: constants.KubeAPIServer, Image: "registry.kl8s.io/kube-apiserver:v1"},
						},
					},
				},
			},
			currentNode:   "node-1",
			expectedNodes: []string{"node-2"},
			expectError:   false,
		},
		{
			name: "one node which is already upgraded",
			pods: []corev1.Pod{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "kube-apiserver-1",
						Namespace: metav1.NamespaceSystem,
						Labels: map[string]string{
							"component": constants.KubeAPIServer,
						},
					},
					Spec: corev1.PodSpec{
						NodeName: "node-1",
						Containers: []corev1.Container{
							{Name: constants.KubeAPIServer, Image: "registry.kl8s.io/kube-apiserver:v2"},
						},
					},
				},
			},
			currentNode:   "node-1",
			expectedNodes: nil,
			expectError:   false,
		},
		{
			name: "two nodes, both already upgraded",
			pods: []corev1.Pod{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "kube-apiserver-1",
						Namespace: metav1.NamespaceSystem,
						Labels: map[string]string{
							"component": constants.KubeAPIServer,
						},
					},
					Spec: corev1.PodSpec{
						NodeName: "node-1",
						Containers: []corev1.Container{
							{Name: constants.KubeAPIServer, Image: "registry.kl8s.io/kube-apiserver:v2"},
						},
					},
				},
				{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "kube-apiserver-2",
						Namespace: metav1.NamespaceSystem,
						Labels: map[string]string{
							"component": constants.KubeAPIServer,
						},
					},
					Spec: corev1.PodSpec{
						NodeName: "node-2",
						Containers: []corev1.Container{
							{Name: constants.KubeAPIServer, Image: "registry.kl8s.io/kube-apiserver:v2"},
						},
					},
				},
			},
			currentNode:   "node-1",
			expectedNodes: nil,
			expectError:   false,
		},
		{
			name:          "no kube-apiserver pods",
			pods:          []corev1.Pod{},
			currentNode:   "node-1",
			expectedNodes: nil,
			expectError:   true,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			var runtimeObjs []runtime.Object
			for _, pod := range tc.pods {
				runtimeObjs = append(runtimeObjs, &pod) // Use pointer
			}
			client := fake.NewSimpleClientset(runtimeObjs...)
			if tc.listErr != nil {
				client.PrependReactor("list", "pods", func(k8stesting.Action) (bool, runtime.Object, error) {
					return true, nil, tc.listErr
				})
			}

			nodes, err := UnupgradedControlPlaneInstances(client, tc.currentNode)
			if tc.expectError != (err != nil) {
				t.Fatalf("expected error: %v, got: %v", tc.expectError, err)
			}

			if !reflect.DeepEqual(nodes, tc.expectedNodes) {
				t.Fatalf("expected unupgraded control plane instances: %v, got: %v", tc.expectedNodes, nodes)
			}
		})
	}
}

// Just some stub code, the code could be enriched when necessary.
type componentConfig struct {
	userSupplied bool
}

func (cc *componentConfig) DeepCopy() kubeadmapi.ComponentConfig {
	result := &componentConfig{}
	return result
}

func (cc *componentConfig) Marshal() ([]byte, error) {
	return nil, nil
}

func (cc *componentConfig) Unmarshal(docmap kubeadmapi.DocumentMap) error {
	return nil
}

func (cc *componentConfig) Get() interface{} {
	return &cc
}

func (cc *componentConfig) Set(cfg interface{}) {
}

func (cc *componentConfig) Default(_ *kubeadmapi.ClusterConfiguration, _ *kubeadmapi.APIEndpoint, _ *kubeadmapi.NodeRegistrationOptions) {
}

func (cc *componentConfig) Mutate() error {
	return nil
}

func (cc *componentConfig) IsUserSupplied() bool {
	return false
}
func (cc *componentConfig) SetUserSupplied(userSupplied bool) {
	cc.userSupplied = userSupplied
}

// moveFiles moves files from one directory to another.
func moveFiles(files map[string]string) error {
	filesToRecover := make(map[string]string, len(files))
	for from, to := range files {
		if err := os.Rename(from, to); err != nil {
			return rollbackFiles(filesToRecover, err)
		}
		filesToRecover[to] = from
	}
	return nil
}

// rollbackFiles moves the files back to the original directory.
func rollbackFiles(files map[string]string, originalErr error) error {
	errs := []error{originalErr}
	for from, to := range files {
		if err := os.Rename(from, to); err != nil {
			errs = append(errs, err)
		}
	}
	return errors.Errorf("couldn't move these files: %v. Got errors: %v", files, errorsutil.NewAggregate(errs))
}
