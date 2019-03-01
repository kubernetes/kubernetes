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
	"reflect"
	"runtime"
	"testing"

	"github.com/pmezard/go-difflib/difflib"

	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeadmapiv1beta1 "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/v1beta1"
	"k8s.io/kubernetes/cmd/kubeadm/app/constants"
)

const (
	controlPlaneV1alpha3YAML          = "testdata/conversion/controlplane/v1alpha3.yaml"
	controlPlaneV1alpha3YAMLNonLinux  = "testdata/conversion/controlplane/v1alpha3_non_linux.yaml"
	controlPlaneV1beta1YAML           = "testdata/conversion/controlplane/v1beta1.yaml"
	controlPlaneV1beta1YAMLNonLinux   = "testdata/conversion/controlplane/v1beta1_non_linux.yaml"
	controlPlaneInternalYAML          = "testdata/conversion/controlplane/internal.yaml"
	controlPlaneInternalYAMLNonLinux  = "testdata/conversion/controlplane/internal_non_linux.yaml"
	controlPlaneIncompleteYAML        = "testdata/defaulting/controlplane/incomplete.yaml"
	controlPlaneDefaultedYAML         = "testdata/defaulting/controlplane/defaulted.yaml"
	controlPlaneDefaultedYAMLNonLinux = "testdata/defaulting/controlplane/defaulted_non_linux.yaml"
	controlPlaneInvalidYAML           = "testdata/validation/invalid_controlplanecfg.yaml"
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
			name:         "v1alpha3.partial1",
			fileContents: cfgFiles["InitConfiguration_v1alpha3"],
			expectErr:    true,
		},
		{
			name:         "v1alpha3.partial2",
			fileContents: cfgFiles["ClusterConfiguration_v1alpha3"],
			expectErr:    true,
		},
		{
			name: "v1alpha3.full",
			fileContents: bytes.Join([][]byte{
				cfgFiles["InitConfiguration_v1alpha3"],
				cfgFiles["ClusterConfiguration_v1alpha3"],
				cfgFiles["Kube-proxy_componentconfig"],
				cfgFiles["Kubelet_componentconfig"],
			}, []byte(constants.YAMLDocumentSeparator)),
			expectErr: true,
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

func TestInitConfigurationMarshallingFromFile(t *testing.T) {
	controlPlaneV1alpha3YAMLAbstracted := controlPlaneV1alpha3YAML
	controlPlaneV1beta1YAMLAbstracted := controlPlaneV1beta1YAML
	controlPlaneInternalYAMLAbstracted := controlPlaneInternalYAML
	controlPlaneDefaultedYAMLAbstracted := controlPlaneDefaultedYAML
	if runtime.GOOS != "linux" {
		controlPlaneV1alpha3YAMLAbstracted = controlPlaneV1alpha3YAMLNonLinux
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
		{ // v1alpha3 -> internal
			name:        "v1alpha3IsDeprecated",
			in:          controlPlaneV1alpha3YAMLAbstracted,
			expectedErr: true,
		},
		{ // v1beta1 -> internal
			name:         "v1beta1ToInternal",
			in:           controlPlaneV1beta1YAMLAbstracted,
			out:          controlPlaneInternalYAMLAbstracted,
			groupVersion: kubeadm.SchemeGroupVersion,
		},
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

func TestConsistentOrderByteSlice(t *testing.T) {
	var (
		aKind = "Akind"
		aFile = []byte(`
kind: Akind
apiVersion: foo.k8s.io/v1
`)
		aaKind = "Aakind"
		aaFile = []byte(`
kind: Aakind
apiVersion: foo.k8s.io/v1
`)
		abKind = "Abkind"
		abFile = []byte(`
kind: Abkind
apiVersion: foo.k8s.io/v1
`)
	)
	var tests = []struct {
		name     string
		in       map[string][]byte
		expected [][]byte
	}{
		{
			name: "a_aa_ab",
			in: map[string][]byte{
				aKind:  aFile,
				aaKind: aaFile,
				abKind: abFile,
			},
			expected: [][]byte{aaFile, abFile, aFile},
		},
		{
			name: "a_ab_aa",
			in: map[string][]byte{
				aKind:  aFile,
				abKind: abFile,
				aaKind: aaFile,
			},
			expected: [][]byte{aaFile, abFile, aFile},
		},
		{
			name: "aa_a_ab",
			in: map[string][]byte{
				aaKind: aaFile,
				aKind:  aFile,
				abKind: abFile,
			},
			expected: [][]byte{aaFile, abFile, aFile},
		},
		{
			name: "aa_ab_a",
			in: map[string][]byte{
				aaKind: aaFile,
				abKind: abFile,
				aKind:  aFile,
			},
			expected: [][]byte{aaFile, abFile, aFile},
		},
		{
			name: "ab_a_aa",
			in: map[string][]byte{
				abKind: abFile,
				aKind:  aFile,
				aaKind: aaFile,
			},
			expected: [][]byte{aaFile, abFile, aFile},
		},
		{
			name: "ab_aa_a",
			in: map[string][]byte{
				abKind: abFile,
				aaKind: aaFile,
				aKind:  aFile,
			},
			expected: [][]byte{aaFile, abFile, aFile},
		},
	}

	for _, rt := range tests {
		t.Run(rt.name, func(t2 *testing.T) {
			actual := consistentOrderByteSlice(rt.in)
			if !reflect.DeepEqual(rt.expected, actual) {
				t2.Errorf("the expected and actual output differs.\n\texpected: %s\n\tout: %s\n", rt.expected, actual)
			}
		})
	}
}
