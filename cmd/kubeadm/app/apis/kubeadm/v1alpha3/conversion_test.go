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

package v1alpha3

import (
	"testing"

	"k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
)

func TestJoinConfigurationConversion(t *testing.T) {
	testcases := map[string]struct {
		old           *JoinConfiguration
		expectedError bool
	}{
		"conversion succeeds": {
			old:           &JoinConfiguration{},
			expectedError: false,
		},
		"cluster name fails to be converted": {
			old: &JoinConfiguration{
				ClusterName: "kubernetes",
			},
			expectedError: true,
		},
		"feature gates fails to be converted": {
			old: &JoinConfiguration{
				FeatureGates: map[string]bool{
					"someGate": true,
				},
			},
			expectedError: true,
		},
	}
	for _, tc := range testcases {
		internal := &kubeadm.JoinConfiguration{}
		err := Convert_v1alpha3_JoinConfiguration_To_kubeadm_JoinConfiguration(tc.old, internal, nil)
		if (err != nil) != tc.expectedError {
			t.Errorf("ImageToImageMeta returned unexpected error: %v, saw: %v", tc.expectedError, (err != nil))
			return
		}
	}
}

func TestInitConfigurationConversion(t *testing.T) {
	testcases := map[string]struct {
		old         *InitConfiguration
		expectedErr bool
	}{
		"conversion succeeds": {
			old:         &InitConfiguration{},
			expectedErr: false,
		},
		"feature gates fails to be converted": {
			old: &InitConfiguration{
				ClusterConfiguration: ClusterConfiguration{
					AuditPolicyConfiguration: AuditPolicyConfiguration{
						Path: "test",
					},
				},
			},
			expectedErr: true,
		},
	}
	for _, tc := range testcases {
		internal := &kubeadm.InitConfiguration{}
		err := Convert_v1alpha3_InitConfiguration_To_kubeadm_InitConfiguration(tc.old, internal, nil)
		if (err != nil) != tc.expectedErr {
			t.Errorf("no error was expected but '%s' was found", err)
		}
	}
}

func TestConvertToUseHyperKubeImage(t *testing.T) {
	tests := []struct {
		desc              string
		in                *ClusterConfiguration
		useHyperKubeImage bool
		expectedErr       bool
	}{
		{
			desc:              "unset UnifiedControlPlaneImage sets UseHyperKubeImage to false",
			in:                &ClusterConfiguration{},
			useHyperKubeImage: false,
			expectedErr:       false,
		},
		{
			desc: "matching UnifiedControlPlaneImage sets UseHyperKubeImage to true",
			in: &ClusterConfiguration{
				ImageRepository:          "k8s.gcr.io",
				KubernetesVersion:        "v1.12.2",
				UnifiedControlPlaneImage: "k8s.gcr.io/hyperkube:v1.12.2",
			},
			useHyperKubeImage: true,
			expectedErr:       false,
		},
		{
			desc: "mismatching UnifiedControlPlaneImage tag causes an error",
			in: &ClusterConfiguration{
				ImageRepository:          "k8s.gcr.io",
				KubernetesVersion:        "v1.12.0",
				UnifiedControlPlaneImage: "k8s.gcr.io/hyperkube:v1.12.2",
			},
			expectedErr: true,
		},
		{
			desc: "mismatching UnifiedControlPlaneImage repo causes an error",
			in: &ClusterConfiguration{
				ImageRepository:          "my.repo",
				KubernetesVersion:        "v1.12.2",
				UnifiedControlPlaneImage: "k8s.gcr.io/hyperkube:v1.12.2",
			},
			expectedErr: true,
		},
		{
			desc: "mismatching UnifiedControlPlaneImage image name causes an error",
			in: &ClusterConfiguration{
				ImageRepository:          "k8s.gcr.io",
				KubernetesVersion:        "v1.12.2",
				UnifiedControlPlaneImage: "k8s.gcr.io/otherimage:v1.12.2",
			},
			expectedErr: true,
		},
	}

	for _, test := range tests {
		t.Run(test.desc, func(t *testing.T) {
			out := &kubeadm.ClusterConfiguration{}
			err := Convert_v1alpha3_UnifiedControlPlaneImage_To_kubeadm_UseHyperKubeImage(test.in, out)
			if test.expectedErr {
				if err == nil {
					t.Fatalf("unexpected success, UseHyperKubeImage: %t", out.UseHyperKubeImage)
				}
			} else {
				if err != nil {
					t.Fatalf("unexpected failure: %v", err)
				}
				if out.UseHyperKubeImage != test.useHyperKubeImage {
					t.Fatalf("mismatching result from conversion:\n\tExpected: %t\n\tReceived: %t", test.useHyperKubeImage, out.UseHyperKubeImage)
				}
			}
		})
	}
}

func TestEtcdImageToImageMeta(t *testing.T) {
	tests := []struct {
		name              string
		image             string
		expectedImageMeta kubeadm.ImageMeta
		expectedError     bool
	}{
		{
			name:  "Empty image -> Empty image meta",
			image: "",
			expectedImageMeta: kubeadm.ImageMeta{
				ImageRepository: "",
				ImageTag:        "",
			},
		},
		{
			name:  "image with tag and repository",
			image: "custom.repo/etcd:custom.tag",
			expectedImageMeta: kubeadm.ImageMeta{
				ImageRepository: "custom.repo",
				ImageTag:        "custom.tag",
			},
		},
		{
			name:          "image with custom imageName",
			image:         "real.repo/custom-image-name-for-etcd:real.tag",
			expectedError: true,
		},
		{
			name:          "image without repository",
			image:         "etcd:real.tag",
			expectedError: true,
		},
		{
			name:          "image without tag",
			image:         "real.repo/etcd",
			expectedError: true,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			ret, err := etcdImageToImageMeta(test.image)

			if (err != nil) != test.expectedError {
				t.Errorf("etcdImageToImageMeta returned unexpected error: %v, saw: %v", test.expectedError, (err != nil))
				return
			}

			if ret != test.expectedImageMeta {
				t.Errorf("etcdImageToImageMeta returned unexpected ImageMeta: %v, saw: %v", test.expectedImageMeta, ret)
			}
		})
	}
}
