/*
Copyright 2016 The Kubernetes Authors.

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

package images

import (
	"fmt"
	"strings"
	"testing"

	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	"k8s.io/kubernetes/cmd/kubeadm/app/constants"
)

const (
	testversion = "v10.1.2-alpha.1.100+0123456789abcdef+SOMETHING"
	expected    = "v10.1.2-alpha.1.100_0123456789abcdef_SOMETHING"
	gcrPrefix   = "k8s.gcr.io"
)

func TestGetGenericImage(t *testing.T) {
	const (
		prefix = "foo"
		image  = "bar"
		tag    = "baz"
	)
	expected := fmt.Sprintf("%s/%s:%s", prefix, image, tag)
	actual := GetGenericImage(prefix, image, tag)
	if actual != expected {
		t.Errorf("failed GetGenericImage:\n\texpected: %s\n\t  actual: %s", expected, actual)
	}
}

func TestGetKubernetesImage(t *testing.T) {
	var tests = []struct {
		image    string
		expected string
		cfg      *kubeadmapi.ClusterConfiguration
	}{
		{
			expected: GetGenericImage(gcrPrefix, constants.HyperKube, expected),
			cfg: &kubeadmapi.ClusterConfiguration{
				ImageRepository:   gcrPrefix,
				KubernetesVersion: testversion,
				UseHyperKubeImage: true,
			},
		},
		{
			image:    constants.KubeAPIServer,
			expected: GetGenericImage(gcrPrefix, "kube-apiserver", expected),
			cfg: &kubeadmapi.ClusterConfiguration{
				ImageRepository:   gcrPrefix,
				KubernetesVersion: testversion,
			},
		},
		{
			image:    constants.KubeControllerManager,
			expected: GetGenericImage(gcrPrefix, "kube-controller-manager", expected),
			cfg: &kubeadmapi.ClusterConfiguration{
				ImageRepository:   gcrPrefix,
				KubernetesVersion: testversion,
			},
		},
		{
			image:    constants.KubeScheduler,
			expected: GetGenericImage(gcrPrefix, "kube-scheduler", expected),
			cfg: &kubeadmapi.ClusterConfiguration{
				ImageRepository:   gcrPrefix,
				KubernetesVersion: testversion,
			},
		},
	}
	for _, rt := range tests {
		actual := GetKubernetesImage(rt.image, rt.cfg)
		if actual != rt.expected {
			t.Errorf(
				"failed GetKubernetesImage:\n\texpected: %s\n\t  actual: %s",
				rt.expected,
				actual,
			)
		}
	}
}

func TestGetEtcdImage(t *testing.T) {
	var tests = []struct {
		expected string
		cfg      *kubeadmapi.ClusterConfiguration
	}{
		{
			cfg: &kubeadmapi.ClusterConfiguration{
				ImageRepository:   "real.repo",
				KubernetesVersion: "1.12.0",
				Etcd: kubeadmapi.Etcd{
					Local: &kubeadmapi.LocalEtcd{},
				},
			},
			expected: "real.repo/etcd:3.2.24",
		},
		{
			cfg: &kubeadmapi.ClusterConfiguration{
				ImageRepository:   "real.repo",
				KubernetesVersion: "1.12.0",
				Etcd: kubeadmapi.Etcd{
					Local: &kubeadmapi.LocalEtcd{
						ImageMeta: kubeadmapi.ImageMeta{
							ImageTag: "override",
						},
					},
				},
			},
			expected: "real.repo/etcd:override",
		},
		{
			cfg: &kubeadmapi.ClusterConfiguration{
				ImageRepository:   "real.repo",
				KubernetesVersion: "1.12.0",
				Etcd: kubeadmapi.Etcd{
					Local: &kubeadmapi.LocalEtcd{
						ImageMeta: kubeadmapi.ImageMeta{
							ImageRepository: "override",
						},
					},
				},
			},
			expected: "override/etcd:3.2.24",
		},
		{
			expected: GetGenericImage(gcrPrefix, "etcd", constants.DefaultEtcdVersion),
			cfg: &kubeadmapi.ClusterConfiguration{
				ImageRepository:   gcrPrefix,
				KubernetesVersion: testversion,
			},
		},
	}
	for _, rt := range tests {
		actual := GetEtcdImage(rt.cfg)
		if actual != rt.expected {
			t.Errorf(
				"failed GetEtcdImage:\n\texpected: %s\n\t  actual: %s",
				rt.expected,
				actual,
			)
		}
	}
}

func TestGetPauseImage(t *testing.T) {
	testcases := []struct {
		name     string
		cfg      *kubeadmapi.ClusterConfiguration
		expected string
	}{
		{
			name: "pause image defined",
			cfg: &kubeadmapi.ClusterConfiguration{
				ImageRepository: "test.repo",
			},
			expected: "test.repo/pause:" + constants.PauseVersion,
		},
	}
	for _, tc := range testcases {
		t.Run(tc.name, func(t *testing.T) {
			actual := GetPauseImage(tc.cfg)
			if actual != tc.expected {
				t.Fatalf(
					"failed GetPauseImage:\n\texpected: %s\n\t  actual: %s",
					tc.expected,
					actual,
				)
			}
		})
	}
}

func TestGetAllImages(t *testing.T) {
	testcases := []struct {
		name   string
		expect string
		cfg    *kubeadmapi.ClusterConfiguration
	}{
		{
			name: "defined CIImageRepository",
			cfg: &kubeadmapi.ClusterConfiguration{
				CIImageRepository: "test.repo",
			},
			expect: "test.repo",
		},
		{
			name: "undefined CIImagerRepository should contain the default image prefix",
			cfg: &kubeadmapi.ClusterConfiguration{
				ImageRepository: "real.repo",
			},
			expect: "real.repo",
		},
		{
			name: "test that etcd is returned when it is not external",
			cfg: &kubeadmapi.ClusterConfiguration{
				Etcd: kubeadmapi.Etcd{
					Local: &kubeadmapi.LocalEtcd{},
				},
			},
			expect: constants.Etcd,
		},
		{
			name: "CoreDNS image is returned",
			cfg: &kubeadmapi.ClusterConfiguration{
				DNS: kubeadmapi.DNS{
					Type: kubeadmapi.CoreDNS,
				},
			},
			expect: constants.CoreDNSImageName,
		},
		{
			name: "main kube-dns image is returned",
			cfg: &kubeadmapi.ClusterConfiguration{
				DNS: kubeadmapi.DNS{
					Type: kubeadmapi.KubeDNS,
				},
			},
			expect: constants.KubeDNSKubeDNSImageName,
		},
		{
			name: "kube-dns sidecar image is returned",
			cfg: &kubeadmapi.ClusterConfiguration{
				DNS: kubeadmapi.DNS{
					Type: kubeadmapi.KubeDNS,
				},
			},
			expect: constants.KubeDNSSidecarImageName,
		},
		{
			name: "kube-dns dnsmasq-nanny image is returned",
			cfg: &kubeadmapi.ClusterConfiguration{
				DNS: kubeadmapi.DNS{
					Type: kubeadmapi.KubeDNS,
				},
			},
			expect: constants.KubeDNSDnsMasqNannyImageName,
		},
	}
	for _, tc := range testcases {
		t.Run(tc.name, func(t *testing.T) {
			imgs := GetControlPlaneImages(tc.cfg)
			for _, img := range imgs {
				if strings.Contains(img, tc.expect) {
					return
				}
			}
			t.Fatalf("did not find %q in %q", tc.expect, imgs)
		})
	}
}
