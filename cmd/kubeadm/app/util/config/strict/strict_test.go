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

package strict

import (
	"io/ioutil"
	"path/filepath"
	"testing"

	"k8s.io/apimachinery/pkg/runtime/schema"
	kubeproxyconfigv1alpha1 "k8s.io/kube-proxy/config/v1alpha1"
	kubeletconfigv1beta1 "k8s.io/kubelet/config/v1beta1"
	kubeadmapiv1beta2 "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/v1beta2"
	"k8s.io/kubernetes/cmd/kubeadm/app/componentconfigs"
	"k8s.io/kubernetes/cmd/kubeadm/app/constants"
)

func TestVerifyUnmarshalStrict(t *testing.T) {
	const (
		pathTestData = "testdata/"
	)

	var testFiles = []struct {
		fileName      string
		kind          string
		groupVersion  schema.GroupVersion
		expectedError bool
	}{
		// tests with file errors
		{
			fileName:      "invalid_duplicate_field_clustercfg.yaml",
			kind:          constants.InitConfigurationKind,
			groupVersion:  kubeadmapiv1beta2.SchemeGroupVersion,
			expectedError: true,
		},
		{
			fileName:      "invalid_duplicate_field_joincfg.yaml",
			kind:          constants.JoinConfigurationKind,
			groupVersion:  kubeadmapiv1beta2.SchemeGroupVersion,
			expectedError: true,
		},
		{
			fileName:      "invalid_duplicate_field_kubeletcfg.yaml",
			kind:          string(componentconfigs.KubeletConfigurationKind),
			groupVersion:  kubeletconfigv1beta1.SchemeGroupVersion,
			expectedError: true,
		},
		{
			fileName:      "invalid_duplicate_field_kubeproxycfg.yaml",
			kind:          string(componentconfigs.KubeProxyConfigurationKind),
			groupVersion:  kubeproxyconfigv1alpha1.SchemeGroupVersion,
			expectedError: true,
		},
		{
			fileName:      "invalid_unknown_field_clustercfg.yaml",
			kind:          constants.ClusterConfigurationKind,
			groupVersion:  kubeadmapiv1beta2.SchemeGroupVersion,
			expectedError: true,
		},
		{
			fileName:      "invalid_unknown_field_initcfg.yaml",
			kind:          constants.InitConfigurationKind,
			groupVersion:  kubeadmapiv1beta2.SchemeGroupVersion,
			expectedError: true,
		},
		{
			fileName:      "invalid_unknown_field_joincfg.yaml",
			kind:          constants.JoinConfigurationKind,
			groupVersion:  kubeadmapiv1beta2.SchemeGroupVersion,
			expectedError: true,
		},
		{
			fileName:      "invalid_unknown_field_kubeletcfg.yaml",
			kind:          string(componentconfigs.KubeletConfigurationKind),
			groupVersion:  kubeletconfigv1beta1.SchemeGroupVersion,
			expectedError: true,
		},
		{
			fileName:      "invalid_unknown_field_kubeproxycfg.yaml",
			kind:          string(componentconfigs.KubeProxyConfigurationKind),
			groupVersion:  kubeproxyconfigv1alpha1.SchemeGroupVersion,
			expectedError: true,
		},
		// test unknown groupVersion and kind
		{
			fileName:      "valid_clustercfg.yaml",
			kind:          constants.ClusterConfigurationKind,
			groupVersion:  schema.GroupVersion{Group: "someGroup", Version: "v1"},
			expectedError: true,
		},
		{
			fileName:      "valid_clustercfg.yaml",
			kind:          "SomeUnknownKind",
			groupVersion:  kubeadmapiv1beta2.SchemeGroupVersion,
			expectedError: true,
		},
		// valid tests
		{
			fileName:      "valid_clustercfg.yaml",
			kind:          constants.ClusterConfigurationKind,
			groupVersion:  kubeadmapiv1beta2.SchemeGroupVersion,
			expectedError: false,
		},
		{
			fileName:      "valid_initcfg.yaml",
			kind:          constants.InitConfigurationKind,
			groupVersion:  kubeadmapiv1beta2.SchemeGroupVersion,
			expectedError: false,
		},
		{
			fileName:      "valid_joincfg.yaml",
			kind:          constants.JoinConfigurationKind,
			groupVersion:  kubeadmapiv1beta2.SchemeGroupVersion,
			expectedError: false,
		},
		{
			fileName:      "valid_kubeletcfg.yaml",
			kind:          string(componentconfigs.KubeletConfigurationKind),
			groupVersion:  kubeletconfigv1beta1.SchemeGroupVersion,
			expectedError: false,
		},
		{
			fileName:      "valid_kubeproxycfg.yaml",
			kind:          string(componentconfigs.KubeProxyConfigurationKind),
			groupVersion:  kubeproxyconfigv1alpha1.SchemeGroupVersion,
			expectedError: false,
		},
	}

	for _, test := range testFiles {
		t.Run(test.fileName, func(t *testing.T) {
			bytes, err := ioutil.ReadFile(filepath.Join(pathTestData, test.fileName))
			if err != nil {
				t.Fatalf("couldn't read test data: %v", err)
			}
			gvk := schema.GroupVersionKind{
				Group:   test.groupVersion.Group,
				Version: test.groupVersion.Version,
				Kind:    test.kind,
			}
			err = VerifyUnmarshalStrict(bytes, gvk)
			if (err != nil) != test.expectedError {
				t.Errorf("expected error %v, got %v, error: %v", err != nil, test.expectedError, err)
			}
		})
	}
}
