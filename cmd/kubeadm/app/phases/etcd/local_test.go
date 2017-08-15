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

package etcd

import (
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"testing"

	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeadmconstants "k8s.io/kubernetes/cmd/kubeadm/app/constants"

	testutil "k8s.io/kubernetes/cmd/kubeadm/test"
)

func TestGetEtcdPodSpec(t *testing.T) {

	// Creates a Master Configuration
	cfg := &kubeadmapi.MasterConfiguration{
		KubernetesVersion: "v1.7.0",
	}

	// Executes GetEtcdPodSpec
	spec := GetEtcdPodSpec(cfg)

	// Assert each specs refers to the right pod
	if spec.Spec.Containers[0].Name != kubeadmconstants.Etcd {
		t.Errorf("getKubeConfigSpecs spec for etcd contains pod %s, expectes %s", spec.Spec.Containers[0].Name, kubeadmconstants.Etcd)
	}
}

func TestCreateLocalEtcdStaticPodManifestFile(t *testing.T) {

	// Create temp folder for the test case
	tmpdir := testutil.SetupTempDir(t)
	defer os.RemoveAll(tmpdir)

	// Creates a Master Configuration
	cfg := &kubeadmapi.MasterConfiguration{
		KubernetesVersion: "v1.7.0",
	}

	// Execute createStaticPodFunction
	manifestPath := filepath.Join(tmpdir, kubeadmconstants.ManifestsSubDirName)
	err := CreateLocalEtcdStaticPodManifestFile(manifestPath, cfg)
	if err != nil {
		t.Errorf("Error executing CreateEtcdStaticPodManifestFile: %v", err)
	}

	// Assert expected files are there
	testutil.AssertFilesCount(t, manifestPath, 1)
	testutil.AssertFileExists(t, manifestPath, kubeadmconstants.Etcd+".yaml")
}

func TestGetEtcdCommand(t *testing.T) {
	var tests = []struct {
		cfg      *kubeadmapi.MasterConfiguration
		expected []string
	}{
		{
			cfg: &kubeadmapi.MasterConfiguration{
				Etcd: kubeadmapi.Etcd{DataDir: "/var/lib/etcd"},
			},
			expected: []string{
				"etcd",
				"--listen-client-urls=http://127.0.0.1:2379",
				"--advertise-client-urls=http://127.0.0.1:2379",
				"--data-dir=/var/lib/etcd",
			},
		},
		{
			cfg: &kubeadmapi.MasterConfiguration{
				Etcd: kubeadmapi.Etcd{
					DataDir: "/var/lib/etcd",
					ExtraArgs: map[string]string{
						"listen-client-urls":    "http://10.0.1.10:2379",
						"advertise-client-urls": "http://10.0.1.10:2379",
					},
				},
			},
			expected: []string{
				"etcd",
				"--listen-client-urls=http://10.0.1.10:2379",
				"--advertise-client-urls=http://10.0.1.10:2379",
				"--data-dir=/var/lib/etcd",
			},
		},
		{
			cfg: &kubeadmapi.MasterConfiguration{
				Etcd: kubeadmapi.Etcd{DataDir: "/etc/foo"},
			},
			expected: []string{
				"etcd",
				"--listen-client-urls=http://127.0.0.1:2379",
				"--advertise-client-urls=http://127.0.0.1:2379",
				"--data-dir=/etc/foo",
			},
		},
	}

	for _, rt := range tests {
		actual := getEtcdCommand(rt.cfg)
		sort.Strings(actual)
		sort.Strings(rt.expected)
		if !reflect.DeepEqual(actual, rt.expected) {
			t.Errorf("failed getEtcdCommand:\nexpected:\n%v\nsaw:\n%v", rt.expected, actual)
		}
	}
}
