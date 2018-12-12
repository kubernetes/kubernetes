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
	kubeadmapiv1alpha3 "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/v1alpha3"
	kubeadmutil "k8s.io/kubernetes/cmd/kubeadm/app/util"
)

const (
	node_v1alpha3YAML = "testdata/conversion/node/v1alpha3.yaml"
	//TODO node_v1beta1YAML   = "testdata/conversion/node/v1beta1.yaml" after introducing v1beta1
	node_internalYAML = "testdata/conversion/node/internal.yaml"
	//TODO node_incompleteYAML = "testdata/defaulting/node/incomplete.yaml" (using v1alpha3) after introducing v1beta1
	node_defaultedYAML = "testdata/defaulting/node/defaulted.yaml"
	node_invalidYAML   = "testdata/validation/invalid_nodecfg.yaml"
)

func TestNodeConfigFileAndDefaultsToInternalConfig(t *testing.T) {
	var tests = []struct {
		name, in, out string
		groupVersion  schema.GroupVersion
		expectedErr   bool
	}{
		// These tests are reading one file, loading it using NodeConfigFileAndDefaultsToInternalConfig that all of kubeadm is using for unmarshal of our API types,
		// and then marshals the internal object to the expected groupVersion
		{ // v1alpha3 -> internal
			name:         "v1alpha3ToInternal",
			in:           node_v1alpha3YAML,
			out:          node_internalYAML,
			groupVersion: kubeadm.SchemeGroupVersion,
		},
		// TODO: implement v1beta1 <-> internal after introducing v1beta1
		// TODO: implement v1alpha3 -> internal -> v1beta1 after introducing v1beta1
		{ // v1alpha3 -> internal -> v1alpha3
			name:         "v1alpha3Tov1alpha3",
			in:           node_v1alpha3YAML,
			out:          node_v1alpha3YAML,
			groupVersion: kubeadmapiv1alpha3.SchemeGroupVersion,
		},
		// These tests are reading one file that has only a subset of the fields populated, loading it using NodeConfigFileAndDefaultsToInternalConfig,
		// and then marshals the internal object to the expected groupVersion
		// TODO: implement v1alpha3 -> default -> validate -> v1beta1 -> v1alpha3 after introducing v1beta1
		{ // v1alpha3 -> validation should fail
			name:        "invalidYAMLShouldFail",
			in:          node_invalidYAML,
			expectedErr: true,
		},
	}

	for _, rt := range tests {
		t.Run(rt.name, func(t2 *testing.T) {

			internalcfg, err := NodeConfigFileAndDefaultsToInternalConfig(rt.in, &kubeadmapiv1alpha3.JoinConfiguration{})
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
