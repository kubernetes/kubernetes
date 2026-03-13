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

package kubelet

import (
	"bytes"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/kubernetes/fake"
	core "k8s.io/client-go/testing"
	kubeletconfig "k8s.io/kubelet/config/v1beta1"
	"k8s.io/utils/ptr"

	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	"k8s.io/kubernetes/cmd/kubeadm/app/componentconfigs"
	configutil "k8s.io/kubernetes/cmd/kubeadm/app/util/config"
)

func TestCreateConfigMap(t *testing.T) {
	nodeName := "fake-node"
	client := fake.NewSimpleClientset()
	client.PrependReactor("get", "nodes", func(action core.Action) (bool, runtime.Object, error) {
		return true, &v1.Node{
			ObjectMeta: metav1.ObjectMeta{
				Name: nodeName,
			},
			Spec: v1.NodeSpec{},
		}, nil
	})
	client.PrependReactor("create", "roles", func(action core.Action) (bool, runtime.Object, error) {
		return true, nil, nil
	})
	client.PrependReactor("create", "rolebindings", func(action core.Action) (bool, runtime.Object, error) {
		return true, nil, nil
	})
	client.PrependReactor("create", "configmaps", func(action core.Action) (bool, runtime.Object, error) {
		return true, nil, nil
	})

	internalcfg, err := configutil.DefaultedStaticInitConfiguration()
	if err != nil {
		t.Fatalf("unexpected failure when defaulting InitConfiguration: %v", err)
	}

	if err := CreateConfigMap(&internalcfg.ClusterConfiguration, client); err != nil {
		t.Errorf("CreateConfigMap: unexpected error %v", err)
	}
}

func TestCreateConfigMapRBACRules(t *testing.T) {
	client := fake.NewSimpleClientset()
	client.PrependReactor("create", "roles", func(action core.Action) (bool, runtime.Object, error) {
		return true, nil, nil
	})
	client.PrependReactor("create", "rolebindings", func(action core.Action) (bool, runtime.Object, error) {
		return true, nil, nil
	})

	if err := createConfigMapRBACRules(client); err != nil {
		t.Errorf("createConfigMapRBACRules: unexpected error %v", err)
	}
}

func TestApplyKubeletConfigPatches(t *testing.T) {
	var (
		input          = []byte("bar: 0\nfoo: 0\n")
		patch          = []byte("bar: 1\n")
		expectedOutput = []byte("bar: 1\nfoo: 0\n")
	)

	dir, err := os.MkdirTemp("", "patches")
	if err != nil {
		t.Fatalf("could not create temp dir: %v", err)
	}
	defer os.RemoveAll(dir)

	if err := os.WriteFile(filepath.Join(dir, "kubeletconfiguration.yaml"), patch, 0644); err != nil {
		t.Fatalf("could not write patch file: %v", err)
	}

	output, err := applyKubeletConfigPatches(input, dir, io.Discard)
	if err != nil {
		t.Fatalf("could not apply patch: %v", err)
	}

	if !bytes.Equal(output, expectedOutput) {
		t.Fatalf("expected output:\n%s\ngot\n%s\n", expectedOutput, output)
	}
}

func TestApplyKubeletConfigPatchFromFile(t *testing.T) {
	const kubeletConfigGVK = "apiVersion: kubelet.config.k8s.io/v1beta1\nkind: KubeletConfiguration\n"

	tests := []struct {
		name           string
		kubeletConfig  []byte
		patchContent   []byte
		expectError    bool
		expectedResult []byte
	}{
		{
			name:           "apply new field",
			kubeletConfig:  []byte(kubeletConfigGVK),
			patchContent:   []byte("containerRuntimeEndpoint: unix:///run/containerd/containerd.sock"),
			expectError:    false,
			expectedResult: []byte("apiVersion: kubelet.config.k8s.io/v1beta1\ncontainerRuntimeEndpoint: unix:///run/containerd/containerd.sock\nkind: KubeletConfiguration\n"),
		},
		{
			name:           "overwrite existing field",
			kubeletConfig:  []byte(kubeletConfigGVK + "containerRuntimeEndpoint: unix:///run/crio/crio.sock\n"),
			patchContent:   []byte("containerRuntimeEndpoint: unix:///run/containerd/containerd.sock"),
			expectError:    false,
			expectedResult: []byte("apiVersion: kubelet.config.k8s.io/v1beta1\ncontainerRuntimeEndpoint: unix:///run/containerd/containerd.sock\nkind: KubeletConfiguration\n"),
		},
		{
			name:          "invalid patch contents",
			kubeletConfig: []byte(kubeletConfigGVK),
			patchContent:  []byte("invalid-patch-content"),
			expectError:   true,
		},
		{
			name:           "empty patch file",
			kubeletConfig:  []byte(kubeletConfigGVK),
			patchContent:   []byte(""),
			expectError:    false,
			expectedResult: []byte(kubeletConfigGVK),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			output := io.Discard

			// Create a temporary file to store the patch content.
			patchFile, err := os.CreateTemp("", "instance-config-*.yml")
			if err != nil {
				t.Errorf("Error creating temporary file: %v", err)
			}
			defer func() {
				_ = patchFile.Close()
				_ = os.Remove(patchFile.Name())
			}()

			_, err = patchFile.Write(tt.patchContent)
			if err != nil {
				t.Errorf("Error writing instance config to file: %v", err)
			}

			// Apply the patch.
			result, err := applyKubeletConfigPatchFromFile(tt.kubeletConfig, patchFile.Name(), output)
			if !tt.expectError && err != nil {
				t.Errorf("Unexpected error: %v", err)
			}
			assert.Equal(t, tt.expectedResult, result)
		})
	}
}

func TestApplyPatchesToConfig(t *testing.T) {
	const (
		expectedAddress = "barfoo"
		expectedPort    = 4321
	)

	kc := &kubeletconfig.KubeletConfiguration{
		HealthzBindAddress: "foobar",
		HealthzPort:        ptr.To[int32](1234),
	}

	cfg := &kubeadmapi.ClusterConfiguration{}
	cfg.ComponentConfigs = kubeadmapi.ComponentConfigMap{}

	localAPIEndpoint := &kubeadmapi.APIEndpoint{}
	nodeRegOps := &kubeadmapi.NodeRegistrationOptions{}
	componentconfigs.Default(cfg, localAPIEndpoint, nodeRegOps)
	cfg.ComponentConfigs[componentconfigs.KubeletGroup].Set(kc)

	// Change to a fake function that does patching with string replace.
	applyKubeletConfigPatchesFunc = func(b []byte, _ string, _ io.Writer) ([]byte, error) {
		b = bytes.ReplaceAll(b, []byte("foobar"), []byte(expectedAddress))
		b = bytes.ReplaceAll(b, []byte("1234"), []byte(fmt.Sprintf("%d", expectedPort)))
		return b, nil
	}
	defer func() {
		applyKubeletConfigPatchesFunc = applyKubeletConfigPatches
	}()

	if err := ApplyPatchesToConfig(cfg, "fakedir"); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	new := cfg.ComponentConfigs[componentconfigs.KubeletGroup].Get()
	newTyped, ok := new.(*kubeletconfig.KubeletConfiguration)
	if !ok {
		t.Fatal("could not cast kubelet config")
	}
	if newTyped.HealthzBindAddress != expectedAddress {
		t.Fatalf("expected address: %s, got: %s", expectedAddress, newTyped.HealthzBindAddress)
	}
	if *newTyped.HealthzPort != expectedPort {
		t.Fatalf("expected port: %d, got: %d", expectedPort, *newTyped.HealthzPort)
	}
}
