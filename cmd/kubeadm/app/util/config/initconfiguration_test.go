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

/*
func TestInitConfigurationMarshallingFromFile(t *testing.T) {
	controlPlaneV1beta1YAMLAbstracted := controlPlaneV1beta1YAML
	controlPlaneInternalYAMLAbstracted := controlPlaneInternalYAML
	controlPlaneDefaultedYAMLAbstracted := controlPlaneDefaultedYAML
	if runtime.GOOS != "linux" {
		controlPlaneV1beta1YAMLAbstracted = controlPlaneV1beta1YAMLNonLinux
		controlPlaneInternalYAMLAbstracted = controlPlaneInternalYAMLNonLinux
		controlPlaneDefaultedYAMLAbstracted = controlPlaneDefaultedYAMLNonLinux
	}

	var tests = []struct {
		name, in, out string
		groupVersion  schema.GroupVersion
		expectedErr   bool
	}{
		// These tests are reading one file, loading it using LoadInitConfigurationFromFile that all of kubeadm is using for unmarshal of our API types,
		// and then marshals the internal object to the expected groupVersion
		//{ // v1beta1 -> internal NB. test commented after changes required for upgrading to go v1.12
		//	name:         "v1beta1ToInternal",
		//	in:           controlPlaneV1beta1YAMLAbstracted,
		//	out:          controlPlaneInternalYAMLAbstracted,
		//	groupVersion: kubeadm.SchemeGroupVersion,
		//},
		{ // v1beta1 -> internal -> v1beta1
			name:         "v1beta1Tov1beta1",
			in:           controlPlaneV1beta1YAMLAbstracted,
			out:          controlPlaneV1beta1YAMLAbstracted,
			groupVersion: kubeadmapiv1beta1.SchemeGroupVersion,
		},
		// These tests are reading one file that has only a subset of the fields populated, loading it using LoadInitConfigurationFromFile,
		// and then marshals the internal object to the expected groupVersion
		{ // v1beta1 -> default -> validate -> internal -> v1beta1
			name:         "incompleteYAMLToDefaultedv1beta1",
			in:           controlPlaneIncompleteYAML,
			out:          controlPlaneDefaultedYAMLAbstracted,
			groupVersion: kubeadmapiv1beta1.SchemeGroupVersion,
		},
		{ // v1beta1 -> validation should fail
			name:        "invalidYAMLShouldFail",
			in:          controlPlaneInvalidYAML,
			expectedErr: true,
		},
	}

	for _, rt := range tests {
		t.Run(rt.name, func(t2 *testing.T) {

			internalcfg, err := LoadInitConfigurationFromFile(rt.in)
			if err != nil {
				if rt.expectedErr {
					return
				}
				t2.Fatalf("couldn't unmarshal test data: %v", err)
			}

			actual, err := MarshalInitConfigurationToBytes(internalcfg, rt.groupVersion)
			if err != nil {
				t2.Fatalf("couldn't marshal internal object: %v", err)
			}

			expected, err := ioutil.ReadFile(rt.out)
			if err != nil {
				t2.Fatalf("couldn't read test data: %v", err)
			}

			if !bytes.Equal(expected, actual) {
				t2.Errorf("the expected and actual output differs.\n\tin: %s\n\tout: %s\n\tgroupversion: %s\n\tdiff: \n%s\n",
					rt.in, rt.out, rt.groupVersion.String(), diff(expected, actual))
			}
		})
	}
}
*/

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
