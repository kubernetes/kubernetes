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
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/pkg/errors"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	errorsutil "k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/client-go/kubernetes/fake"

	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	"k8s.io/kubernetes/cmd/kubeadm/app/componentconfigs"
	"k8s.io/kubernetes/cmd/kubeadm/app/constants"
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
	tempDir := t.TempDir()
	testCases := []struct {
		name          string
		patchesDir    string
		expectedError bool
		cfg           *kubeadmapi.InitConfiguration
	}{
		{
			name: "write kubelet config file successfully",
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
			expectedError: true,
			cfg:           &kubeadmapi.InitConfiguration{},
		},
		{
			name:          "only one err: patch dir does not exist",
			patchesDir:    "Bogus",
			expectedError: true,
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
		err := WriteKubeletConfigFiles(tc.cfg, tempDir, tc.patchesDir, true, os.Stdout)
		if (err != nil) != tc.expectedError {
			t.Fatalf("expected error: %v, got: %v, error: %v", tc.expectedError, err != nil, err)
		}
	}
}

func TestUnupgradedControlPlaneInstances(t *testing.T) {
	testCases := []struct {
		name          string
		pods          []corev1.Pod
		currentNode   string
		expectedNodes []string
		expectError   bool
	}{
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
