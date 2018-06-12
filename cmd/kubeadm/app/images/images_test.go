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
	"runtime"
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

func TestGetCoreImage(t *testing.T) {
	var tests = []struct {
		image, repo, version, override, expected string
	}{
		{
			override: "override",
			expected: "override",
		},
		{
			image:    constants.Etcd,
			repo:     gcrPrefix,
			expected: fmt.Sprintf("%s/%s-%s:%s", gcrPrefix, "etcd", runtime.GOARCH, constants.DefaultEtcdVersion),
		},
		{
			image:    constants.KubeAPIServer,
			repo:     gcrPrefix,
			version:  testversion,
			expected: fmt.Sprintf("%s/%s-%s:%s", gcrPrefix, "kube-apiserver", runtime.GOARCH, expected),
		},
		{
			image:    constants.KubeControllerManager,
			repo:     gcrPrefix,
			version:  testversion,
			expected: fmt.Sprintf("%s/%s-%s:%s", gcrPrefix, "kube-controller-manager", runtime.GOARCH, expected),
		},
		{
			image:    constants.KubeScheduler,
			repo:     gcrPrefix,
			version:  testversion,
			expected: fmt.Sprintf("%s/%s-%s:%s", gcrPrefix, "kube-scheduler", runtime.GOARCH, expected),
		},
	}
	for _, rt := range tests {
		actual := GetCoreImage(rt.image, rt.repo, rt.version, rt.override)
		if actual != rt.expected {
			t.Errorf(
				"failed GetCoreImage:\n\texpected: %s\n\t  actual: %s",
				rt.expected,
				actual,
			)
		}
	}
}

func TestGetAllImages(t *testing.T) {
	testcases := []struct {
		name   string
		cfg    *kubeadmapi.MasterConfiguration
		expect string
	}{
		{
			name: "defined CIImageRepository",
			cfg: &kubeadmapi.MasterConfiguration{
				CIImageRepository: "test.repo",
			},
			expect: "test.repo",
		},
		{
			name: "undefined CIImagerRepository should contain the default image prefix",
			cfg: &kubeadmapi.MasterConfiguration{
				ImageRepository: "real.repo",
			},
			expect: "real.repo",
		},
		{
			name: "test that etcd is returned when it is not external",
			cfg: &kubeadmapi.MasterConfiguration{
				Etcd: kubeadmapi.Etcd{
					Local: &kubeadmapi.LocalEtcd{},
				},
			},
			expect: constants.Etcd,
		},
	}
	for _, tc := range testcases {
		t.Run(tc.name, func(t *testing.T) {
			imgs := GetAllImages(tc.cfg)
			for _, img := range imgs {
				if strings.Contains(img, tc.expect) {
					return
				}
			}
			t.Fatalf("did not find %q in %q", tc.expect, imgs)
		})
	}
}
