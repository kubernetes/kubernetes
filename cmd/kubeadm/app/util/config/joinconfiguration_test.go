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
	"testing"

	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	"k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/scheme"
	kubeadmapiv1beta1 "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/v1beta1"
	kubeadmapiv1beta2 "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/v1beta2"
	kubeadmutil "k8s.io/kubernetes/cmd/kubeadm/app/util"
)

const (
	nodeV1beta1YAML    = "testdata/conversion/node/v1beta1.yaml"
	nodeV1beta2YAML    = "testdata/conversion/node/v1beta2.yaml"
	nodeInternalYAML   = "testdata/conversion/node/internal.yaml"
	nodeIncompleteYAML = "testdata/defaulting/node/incomplete.yaml"
	nodeDefaultedYAML  = "testdata/defaulting/node/defaulted.yaml"
	nodeInvalidYAML    = "testdata/validation/invalid_nodecfg.yaml"
)

func TestLoadJoinConfigurationFromFile(t *testing.T) {
	var tests = []struct {
		name, in, out string
		groupVersion  schema.GroupVersion
		expectedErr   bool
	}{
		// These tests are reading one file, loading it using LoadJoinConfigurationFromFile that all of kubeadm is using for unmarshal of our API types,
		// and then marshals the internal object to the expected groupVersion
		{ // v1beta1 -> internal
			name:         "v1beta1ToInternal",
			in:           nodeV1beta1YAML,
			out:          nodeInternalYAML,
			groupVersion: kubeadm.SchemeGroupVersion,
		},
		{ // v1beta1 -> internal -> v1beta1
			name:         "v1beta1Tov1beta1",
			in:           nodeV1beta1YAML,
			out:          nodeV1beta1YAML,
			groupVersion: kubeadmapiv1beta1.SchemeGroupVersion,
		},
		{ // v1beta2 -> internal
			name:         "v1beta2ToInternal",
			in:           nodeV1beta2YAML,
			out:          nodeInternalYAML,
			groupVersion: kubeadm.SchemeGroupVersion,
		},
		{ // v1beta2 -> internal -> v1beta2
			name:         "v1beta2Tov1beta2",
			in:           nodeV1beta2YAML,
			out:          nodeV1beta2YAML,
			groupVersion: kubeadmapiv1beta2.SchemeGroupVersion,
		},
		// These tests are reading one file that has only a subset of the fields populated, loading it using LoadJoinConfigurationFromFile,
		// and then marshals the internal object to the expected groupVersion
		{ // v1beta2 -> default -> validate -> internal -> v1beta2
			name:         "incompleteYAMLToDefaultedv1beta2",
			in:           nodeIncompleteYAML,
			out:          nodeDefaultedYAML,
			groupVersion: kubeadmapiv1beta2.SchemeGroupVersion,
		},
		{ // v1beta2 -> validation should fail
			name:        "invalidYAMLShouldFail",
			in:          nodeInvalidYAML,
			expectedErr: true,
		},
	}

	for _, rt := range tests {
		t.Run(rt.name, func(t2 *testing.T) {

			internalcfg, err := LoadJoinConfigurationFromFile(rt.in)
			if err != nil {
				if rt.expectedErr {
					return
				}
				t2.Fatalf("couldn't unmarshal test data: %v", err)
			}

			actual, err := kubeadmutil.MarshalToYamlForCodecs(internalcfg, rt.groupVersion, scheme.Codecs)
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
