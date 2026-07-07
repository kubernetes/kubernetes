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
	"testing"

	"github.com/stretchr/testify/assert"

	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeadmapiv1 "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/v1beta4"
	"k8s.io/kubernetes/cmd/kubeadm/app/constants"
)

const (
	testVersion = "v1.99.0"
	gcrPrefix   = "registry.k8s.io"
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
			image:    constants.KubeAPIServer,
			expected: GetGenericImage(gcrPrefix, "kube-apiserver", testVersion),
			cfg: &kubeadmapi.ClusterConfiguration{
				ImageRepository:   gcrPrefix,
				KubernetesVersion: testVersion,
			},
		},
		{
			image:    constants.KubeControllerManager,
			expected: GetGenericImage(gcrPrefix, "kube-controller-manager", testVersion),
			cfg: &kubeadmapi.ClusterConfiguration{
				ImageRepository:   gcrPrefix,
				KubernetesVersion: testVersion,
			},
		},
		{
			image:    constants.KubeScheduler,
			expected: GetGenericImage(gcrPrefix, "kube-scheduler", testVersion),
			cfg: &kubeadmapi.ClusterConfiguration{
				ImageRepository:   gcrPrefix,
				KubernetesVersion: testVersion,
			},
		},
		{
			image:    constants.KubeProxy,
			expected: GetGenericImage(gcrPrefix, "kube-proxy", testVersion),
			cfg: &kubeadmapi.ClusterConfiguration{
				ImageRepository:   gcrPrefix,
				KubernetesVersion: testVersion,
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
	testEtcdVer, _, _ := constants.EtcdSupportedVersion(constants.SupportedEtcdVersion, testVersion)
	var tests = []struct {
		expected string
		cfg      *kubeadmapi.ClusterConfiguration
	}{
		{
			cfg: &kubeadmapi.ClusterConfiguration{
				ImageRepository:   "real.repo",
				KubernetesVersion: testVersion,
				Etcd: kubeadmapi.Etcd{
					Local: &kubeadmapi.LocalEtcd{},
				},
			},
			expected: "real.repo/etcd:" + testEtcdVer.String(),
		},
		{
			cfg: &kubeadmapi.ClusterConfiguration{
				ImageRepository:   "real.repo",
				KubernetesVersion: testVersion,
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
				KubernetesVersion: testVersion,
				Etcd: kubeadmapi.Etcd{
					Local: &kubeadmapi.LocalEtcd{
						ImageMeta: kubeadmapi.ImageMeta{
							ImageRepository: "override",
						},
					},
				},
			},
			expected: "override/etcd:" + testEtcdVer.String(),
		},
		{
			expected: GetGenericImage(gcrPrefix, "etcd", constants.DefaultEtcdVersion),
			cfg: &kubeadmapi.ClusterConfiguration{
				ImageRepository:   gcrPrefix,
				KubernetesVersion: testVersion,
			},
		},
	}
	for _, rt := range tests {
		actual := GetEtcdImage(rt.cfg, constants.SupportedEtcdVersion)
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
		name           string
		expectedImages []string
		cfg            *kubeadmapi.ClusterConfiguration
	}{
		{
			name: "defined CIImageRepository",
			cfg: &kubeadmapi.ClusterConfiguration{
				CIImageRepository: "test.repo",
			},
			expectedImages: []string{
				"test.repo/kube-apiserver:",
				"test.repo/kube-controller-manager:",
				"test.repo/kube-scheduler:",
				"test.repo/kube-proxy:",
				"/coredns:" + constants.DefaultCoreDNSVersion,
				"/pause:" + constants.PauseVersion,
			},
		},
		{
			name: "undefined CIImagerRepository should contain the default image prefix",
			cfg: &kubeadmapi.ClusterConfiguration{
				ImageRepository: "real.repo",
			},
			expectedImages: []string{
				"real.repo/kube-apiserver:",
				"real.repo/kube-controller-manager:",
				"real.repo/kube-scheduler:",
				"real.repo/kube-proxy:",
				"real.repo/coredns:" + constants.DefaultCoreDNSVersion,
				"real.repo/pause:" + constants.PauseVersion,
			},
		},
		{
			name: "test that etcd is returned when it is not external",
			cfg: &kubeadmapi.ClusterConfiguration{
				Etcd: kubeadmapi.Etcd{
					Local: &kubeadmapi.LocalEtcd{},
				},
			},
			expectedImages: []string{
				"/kube-apiserver:",
				"/kube-controller-manager:",
				"/kube-scheduler:",
				"/kube-proxy:",
				"/coredns:" + constants.DefaultCoreDNSVersion,
				"/pause:" + constants.PauseVersion,
				"/etcd:" + constants.DefaultEtcdVersion,
			},
		},
		{
			name: "CoreDNS and kube-proxy image are returned",
			cfg:  &kubeadmapi.ClusterConfiguration{},
			expectedImages: []string{
				"/kube-apiserver:",
				"/kube-controller-manager:",
				"/kube-scheduler:",
				"/kube-proxy:",
				"/coredns:" + constants.DefaultCoreDNSVersion,
				"/pause:" + constants.PauseVersion,
			},
		},
		{
			name: "CoreDNS image is skipped",
			cfg: &kubeadmapi.ClusterConfiguration{
				DNS: kubeadmapi.DNS{
					Disabled: true,
				},
			},
			expectedImages: []string{
				"/kube-apiserver:",
				"/kube-controller-manager:",
				"/kube-scheduler:",
				"/kube-proxy:",
				"/pause:" + constants.PauseVersion,
			},
		},
		{
			name: "kube-proxy image is skipped",
			cfg: &kubeadmapi.ClusterConfiguration{
				Proxy: kubeadmapi.Proxy{
					Disabled: true,
				},
			},
			expectedImages: []string{
				"/kube-apiserver:",
				"/kube-controller-manager:",
				"/kube-scheduler:",
				"/coredns:" + constants.DefaultCoreDNSVersion,
				"/pause:" + constants.PauseVersion,
			},
		},
		{
			name: "setting addons Disabled to false has no effect",
			cfg: &kubeadmapi.ClusterConfiguration{
				DNS: kubeadmapi.DNS{
					Disabled: false,
				},
				Proxy: kubeadmapi.Proxy{
					Disabled: false,
				},
			},
			expectedImages: []string{
				"/kube-apiserver:",
				"/kube-controller-manager:",
				"/kube-scheduler:",
				"/kube-proxy:",
				"/coredns:" + constants.DefaultCoreDNSVersion,
				"/pause:" + constants.PauseVersion,
			},
		},
	}
	for _, tc := range testcases {
		t.Run(tc.name, func(t *testing.T) {
			imgs := GetControlPlaneImages(tc.cfg)
			assert.Equal(t, tc.expectedImages, imgs)
		})
	}
}

func TestGetDNSImage(t *testing.T) {
	var tests = []struct {
		expected string
		cfg      *kubeadmapi.ClusterConfiguration
	}{
		{
			expected: "foo.io/coredns:" + constants.DefaultCoreDNSVersion,
			cfg: &kubeadmapi.ClusterConfiguration{
				ImageRepository: "foo.io",
				DNS:             kubeadmapi.DNS{},
			},
		},
		{
			expected: kubeadmapiv1.DefaultImageRepository + "/coredns/coredns:" + constants.DefaultCoreDNSVersion,
			cfg: &kubeadmapi.ClusterConfiguration{
				ImageRepository: kubeadmapiv1.DefaultImageRepository,
				DNS:             kubeadmapi.DNS{},
			},
		},
		{
			expected: "foo.io/coredns/coredns:" + constants.DefaultCoreDNSVersion,
			cfg: &kubeadmapi.ClusterConfiguration{
				ImageRepository: "foo.io",
				DNS: kubeadmapi.DNS{
					ImageMeta: kubeadmapi.ImageMeta{
						ImageRepository: "foo.io/coredns",
					},
				},
			},
		},
		{
			expected: "foo.io/coredns/coredns:" + constants.DefaultCoreDNSVersion,
			cfg: &kubeadmapi.ClusterConfiguration{
				ImageRepository: "foo.io/coredns",
				DNS: kubeadmapi.DNS{
					ImageMeta: kubeadmapi.ImageMeta{
						ImageTag: constants.DefaultCoreDNSVersion,
					},
				},
			},
		},
		{
			expected: "foo.io/coredns:v9.9.9",
			cfg: &kubeadmapi.ClusterConfiguration{
				ImageRepository: "foo.io",
				DNS: kubeadmapi.DNS{
					ImageMeta: kubeadmapi.ImageMeta{
						ImageTag: "v9.9.9",
					},
				},
			},
		},
	}

	for _, test := range tests {
		actual := GetDNSImage(test.cfg)
		if actual != test.expected {
			t.Errorf(
				"failed to GetDNSImage:\n\texpected: %s\n\t actual: %s",
				test.expected,
				actual,
			)
		}
	}
}

func TestGetDNSImageTag(t *testing.T) {
	supportedCoreDNSVersion := map[uint8]string{
		17: "v1.6.7",
		18: "v1.7.0",
	}
	var tests = []struct {
		name     string
		expected string
		cfg      *kubeadmapi.ClusterConfiguration
	}{
		{
			name:     "resolves the version matching the Kubernetes minor",
			expected: "v1.6.7",
			cfg:      &kubeadmapi.ClusterConfiguration{KubernetesVersion: "1.17.0"},
		},
		{
			name:     "falls back to the nearest version when the Kubernetes minor is out of range",
			expected: "v1.7.0",
			cfg:      &kubeadmapi.ClusterConfiguration{KubernetesVersion: "1.99.0"},
		},
		{
			name:     "user override wins over the resolved version",
			expected: "v9.9.9",
			cfg: &kubeadmapi.ClusterConfiguration{
				KubernetesVersion: "1.17.0",
				DNS: kubeadmapi.DNS{
					ImageMeta: kubeadmapi.ImageMeta{ImageTag: "v9.9.9"},
				},
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			actual := GetDNSImageTag(test.cfg, supportedCoreDNSVersion)
			if actual != test.expected {
				t.Errorf("failed to GetDNSImageTag:\n\texpected: %s\n\tactual: %s", test.expected, actual)
			}
		})
	}
}
