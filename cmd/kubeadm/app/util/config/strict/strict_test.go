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

package strict_test

import (
	"os"
	"path/filepath"
	"testing"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	kubeproxyconfigv1alpha1 "k8s.io/kube-proxy/config/v1alpha1"
	kubeletconfigv1beta1 "k8s.io/kubelet/config/v1beta1"

	kubeadmscheme "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/scheme"
	kubeadmapiv1 "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/v1beta4"
	"k8s.io/kubernetes/cmd/kubeadm/app/componentconfigs"
	"k8s.io/kubernetes/cmd/kubeadm/app/constants"
	"k8s.io/kubernetes/cmd/kubeadm/app/util/config/strict"
)

func TestVerifyUnmarshalStrict(t *testing.T) {
	const (
		pathTestData = "testdata/"
	)

	var schemes = []*runtime.Scheme{kubeadmscheme.Scheme, componentconfigs.Scheme}
	var testFiles = []struct {
		fileName      string
		kind          string
		groupVersion  schema.GroupVersion
		expectedError bool
	}{
		// tests with file errors
		{
			fileName:      "invalid_casesensitive_field.yaml",
			kind:          constants.ClusterConfigurationKind,
			groupVersion:  kubeadmapiv1.SchemeGroupVersion,
			expectedError: true,
		},
		{
			fileName:      "invalid_duplicate_field_clustercfg.yaml",
			kind:          constants.InitConfigurationKind,
			groupVersion:  kubeadmapiv1.SchemeGroupVersion,
			expectedError: true,
		},
		{
			fileName:      "invalid_duplicate_field_joincfg.yaml",
			kind:          constants.JoinConfigurationKind,
			groupVersion:  kubeadmapiv1.SchemeGroupVersion,
			expectedError: true,
		},
		{
			fileName:      "invalid_duplicate_field_kubeletcfg.yaml",
			kind:          "KubeletConfiguration",
			groupVersion:  kubeletconfigv1beta1.SchemeGroupVersion,
			expectedError: true,
		},
		{
			fileName:      "invalid_duplicate_field_kubeproxycfg.yaml",
			kind:          "KubeProxyConfiguration",
			groupVersion:  kubeproxyconfigv1alpha1.SchemeGroupVersion,
			expectedError: true,
		},
		{
			fileName:      "invalid_unknown_field_clustercfg.yaml",
			kind:          constants.ClusterConfigurationKind,
			groupVersion:  kubeadmapiv1.SchemeGroupVersion,
			expectedError: true,
		},
		{
			fileName:      "invalid_unknown_field_initcfg.yaml",
			kind:          constants.InitConfigurationKind,
			groupVersion:  kubeadmapiv1.SchemeGroupVersion,
			expectedError: true,
		},
		{
			fileName:      "invalid_unknown_field_joincfg.yaml",
			kind:          constants.JoinConfigurationKind,
			groupVersion:  kubeadmapiv1.SchemeGroupVersion,
			expectedError: true,
		},
		{
			fileName:      "invalid_unknown_field_kubeletcfg.yaml",
			kind:          "KubeletConfiguration",
			groupVersion:  kubeletconfigv1beta1.SchemeGroupVersion,
			expectedError: true,
		},
		{
			fileName:      "invalid_unknown_field_kubeproxycfg.yaml",
			kind:          "KubeProxyConfiguration",
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
			groupVersion:  kubeadmapiv1.SchemeGroupVersion,
			expectedError: true,
		},
		// valid tests
		{
			fileName:      "valid_clustercfg.yaml",
			kind:          constants.ClusterConfigurationKind,
			groupVersion:  kubeadmapiv1.SchemeGroupVersion,
			expectedError: false,
		},
		{
			fileName:      "valid_initcfg.yaml",
			kind:          constants.InitConfigurationKind,
			groupVersion:  kubeadmapiv1.SchemeGroupVersion,
			expectedError: false,
		},
		{
			fileName:      "valid_joincfg.yaml",
			kind:          constants.JoinConfigurationKind,
			groupVersion:  kubeadmapiv1.SchemeGroupVersion,
			expectedError: false,
		},
		{
			fileName:      "valid_kubeletcfg.yaml",
			kind:          "KubeletConfiguration",
			groupVersion:  kubeletconfigv1beta1.SchemeGroupVersion,
			expectedError: false,
		},
		{
			fileName:      "valid_kubeproxycfg.yaml",
			kind:          "KubeProxyConfiguration",
			groupVersion:  kubeproxyconfigv1alpha1.SchemeGroupVersion,
			expectedError: false,
		},
	}

	for _, test := range testFiles {
		t.Run(test.fileName, func(t *testing.T) {
			bytes, err := os.ReadFile(filepath.Join(pathTestData, test.fileName))
			if err != nil {
				t.Fatalf("couldn't read test data: %v", err)
			}
			gvk := schema.GroupVersionKind{
				Group:   test.groupVersion.Group,
				Version: test.groupVersion.Version,
				Kind:    test.kind,
			}
			err = strict.VerifyUnmarshalStrict(schemes, gvk, bytes)
			if (err != nil) != test.expectedError {
				t.Errorf("expected error %v, got %v, error: %v", err != nil, test.expectedError, err)
			}
		})
	}
}
