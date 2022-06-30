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
	"io/ioutil"
	"os"
	"path/filepath"
	"testing"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/kubernetes/fake"
	core "k8s.io/client-go/testing"

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

	dir, err := ioutil.TempDir("", "patches")
	if err != nil {
		t.Fatalf("could not create temp dir: %v", err)
	}
	defer os.RemoveAll(dir)

	if err := ioutil.WriteFile(filepath.Join(dir, "kubeletconfiguration.yaml"), patch, 0644); err != nil {
		t.Fatalf("could not write patch file: %v", err)
	}

	output, err := applyKubeletConfigPatches(input, dir, ioutil.Discard)
	if err != nil {
		t.Fatalf("could not apply patch: %v", err)
	}

	if !bytes.Equal(output, expectedOutput) {
		t.Fatalf("expected output:\n%s\ngot\n%s\n", expectedOutput, output)
	}
}
