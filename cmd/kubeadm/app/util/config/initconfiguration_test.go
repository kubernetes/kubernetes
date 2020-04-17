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

package config

import (
	"bytes"
	"io/ioutil"
	"os"
	"path/filepath"
	"testing"

	"github.com/pmezard/go-difflib/difflib"

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	kubeadmapiv1beta2 "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/v1beta2"
	"k8s.io/kubernetes/cmd/kubeadm/app/constants"
	"sigs.k8s.io/yaml"
)

func diff(expected, actual []byte) string {
	// Write out the diff
	var diffBytes bytes.Buffer
	difflib.WriteUnifiedDiff(&diffBytes, difflib.UnifiedDiff{
		A:        difflib.SplitLines(string(expected)),
		B:        difflib.SplitLines(string(actual)),
		FromFile: "expected",
		ToFile:   "actual",
		Context:  3,
	})
	return diffBytes.String()
}

func TestLoadInitConfigurationFromFile(t *testing.T) {
	// Create temp folder for the test case
	tmpdir, err := ioutil.TempDir("", "")
	if err != nil {
		t.Fatalf("Couldn't create tmpdir")
	}
	defer os.RemoveAll(tmpdir)

	// cfgFiles is in cluster_test.go
	var tests = []struct {
		name         string
		fileContents []byte
		expectErr    bool
	}{
		{
			name:         "v1beta1.partial1",
			fileContents: cfgFiles["InitConfiguration_v1beta1"],
		},
		{
			name:         "v1beta1.partial2",
			fileContents: cfgFiles["ClusterConfiguration_v1beta1"],
		},
		{
			name: "v1beta1.full",
			fileContents: bytes.Join([][]byte{
				cfgFiles["InitConfiguration_v1beta1"],
				cfgFiles["ClusterConfiguration_v1beta1"],
				cfgFiles["Kube-proxy_componentconfig"],
				cfgFiles["Kubelet_componentconfig"],
			}, []byte(constants.YAMLDocumentSeparator)),
		},
		{
			name:         "v1beta2.partial1",
			fileContents: cfgFiles["InitConfiguration_v1beta2"],
		},
		{
			name:         "v1beta2.partial2",
			fileContents: cfgFiles["ClusterConfiguration_v1beta2"],
		},
		{
			name: "v1beta2.full",
			fileContents: bytes.Join([][]byte{
				cfgFiles["InitConfiguration_v1beta2"],
				cfgFiles["ClusterConfiguration_v1beta2"],
				cfgFiles["Kube-proxy_componentconfig"],
				cfgFiles["Kubelet_componentconfig"],
			}, []byte(constants.YAMLDocumentSeparator)),
		},
	}

	for _, rt := range tests {
		t.Run(rt.name, func(t2 *testing.T) {
			cfgPath := filepath.Join(tmpdir, rt.name)
			err := ioutil.WriteFile(cfgPath, rt.fileContents, 0644)
			if err != nil {
				t.Errorf("Couldn't create file")
				return
			}

			obj, err := LoadInitConfigurationFromFile(cfgPath)
			if rt.expectErr {
				if err == nil {
					t.Error("Unexpected success")
				}
			} else {
				if err != nil {
					t.Errorf("Error reading file: %v", err)
					return
				}

				if obj == nil {
					t.Errorf("Unexpected nil return value")
				}
			}
		})
	}
}

func TestDefaultTaintsMarshaling(t *testing.T) {
	tests := []struct {
		desc             string
		cfg              kubeadmapiv1beta2.InitConfiguration
		expectedTaintCnt int
	}{
		{
			desc: "Uninitialized nodeRegistration field produces a single taint (the master one)",
			cfg: kubeadmapiv1beta2.InitConfiguration{
				TypeMeta: metav1.TypeMeta{
					APIVersion: "kubeadm.k8s.io/v1beta2",
					Kind:       constants.InitConfigurationKind,
				},
			},
			expectedTaintCnt: 1,
		},
		{
			desc: "Uninitialized taints field produces a single taint (the master one)",
			cfg: kubeadmapiv1beta2.InitConfiguration{
				TypeMeta: metav1.TypeMeta{
					APIVersion: "kubeadm.k8s.io/v1beta2",
					Kind:       constants.InitConfigurationKind,
				},
				NodeRegistration: kubeadmapiv1beta2.NodeRegistrationOptions{},
			},
			expectedTaintCnt: 1,
		},
		{
			desc: "Forsing taints to an empty slice produces no taints",
			cfg: kubeadmapiv1beta2.InitConfiguration{
				TypeMeta: metav1.TypeMeta{
					APIVersion: "kubeadm.k8s.io/v1beta2",
					Kind:       constants.InitConfigurationKind,
				},
				NodeRegistration: kubeadmapiv1beta2.NodeRegistrationOptions{
					Taints: []v1.Taint{},
				},
			},
			expectedTaintCnt: 0,
		},
		{
			desc: "Custom taints are used",
			cfg: kubeadmapiv1beta2.InitConfiguration{
				TypeMeta: metav1.TypeMeta{
					APIVersion: "kubeadm.k8s.io/v1beta2",
					Kind:       constants.InitConfigurationKind,
				},
				NodeRegistration: kubeadmapiv1beta2.NodeRegistrationOptions{
					Taints: []v1.Taint{
						{Key: "taint1"},
						{Key: "taint2"},
					},
				},
			},
			expectedTaintCnt: 2,
		},
	}

	for _, tc := range tests {
		t.Run(tc.desc, func(t *testing.T) {
			b, err := yaml.Marshal(tc.cfg)
			if err != nil {
				t.Fatalf("unexpected error while marshalling to YAML: %v", err)
			}

			cfg, err := BytesToInitConfiguration(b)
			if err != nil {
				t.Fatalf("unexpected error of BytesToInitConfiguration: %v\nconfig: %s", err, string(b))
			}

			if tc.expectedTaintCnt != len(cfg.NodeRegistration.Taints) {
				t.Fatalf("unexpected taints count\nexpected: %d\ngot: %d\ntaints: %v", tc.expectedTaintCnt, len(cfg.NodeRegistration.Taints), cfg.NodeRegistration.Taints)
			}
		})
	}
}
