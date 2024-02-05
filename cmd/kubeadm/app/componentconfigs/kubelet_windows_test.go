/*
Copyright 2021 The Kubernetes Authors.

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

package componentconfigs

import (
	"path/filepath"
	"testing"

	"github.com/google/go-cmp/cmp"

	kubeletconfig "k8s.io/kubelet/config/v1beta1"
	"k8s.io/utils/ptr"
)

func TestMutatePaths(t *testing.T) {
	const drive = "C:"
	var fooResolverConfig string = "/foo/resolver"

	tests := []struct {
		name     string
		cfg      *kubeletconfig.KubeletConfiguration
		expected *kubeletconfig.KubeletConfiguration
	}{
		{
			name: "valid: all fields are absolute paths",
			cfg: &kubeletconfig.KubeletConfiguration{
				ResolverConfig: &fooResolverConfig,
				StaticPodPath:  "/foo/staticpods",
				Authentication: kubeletconfig.KubeletAuthentication{
					X509: kubeletconfig.KubeletX509Authentication{
						ClientCAFile: "/foo/ca.crt",
					},
				},
			},
			expected: &kubeletconfig.KubeletConfiguration{
				ResolverConfig: ptr.To(""),
				StaticPodPath:  filepath.Join(drive, "/foo/staticpods"),
				Authentication: kubeletconfig.KubeletAuthentication{
					X509: kubeletconfig.KubeletX509Authentication{
						ClientCAFile: filepath.Join(drive, "/foo/ca.crt"),
					},
				},
			},
		},
		{
			name: "valid: some fields are not absolute paths",
			cfg: &kubeletconfig.KubeletConfiguration{
				ResolverConfig: &fooResolverConfig,
				StaticPodPath:  "./foo/staticpods", // not an absolute Unix path
				Authentication: kubeletconfig.KubeletAuthentication{
					X509: kubeletconfig.KubeletX509Authentication{
						ClientCAFile: "/foo/ca.crt",
					},
				},
			},
			expected: &kubeletconfig.KubeletConfiguration{
				ResolverConfig: ptr.To(""),
				StaticPodPath:  "./foo/staticpods",
				Authentication: kubeletconfig.KubeletAuthentication{
					X509: kubeletconfig.KubeletX509Authentication{
						ClientCAFile: filepath.Join(drive, "/foo/ca.crt"),
					},
				},
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			mutatePaths(test.cfg, drive)
			if diff := cmp.Diff(test.cfg, test.expected); len(diff) > 0 {
				t.Errorf("Mismatch between expected (-) and got (+):\n%s", diff)
			}
		})
	}
}

func TestMutateDefaults(t *testing.T) {
	tests := []struct {
		name     string
		cfg      *kubeletconfig.KubeletConfiguration
		expected *kubeletconfig.KubeletConfiguration
	}{
		{
			name: "fields of interest get mutated",
			cfg: &kubeletconfig.KubeletConfiguration{
				EnforceNodeAllocatable: []string{"pods"},
				CgroupsPerQOS:          ptr.To(true),
			},
			expected: &kubeletconfig.KubeletConfiguration{
				EnforceNodeAllocatable: []string{},
				CgroupsPerQOS:          ptr.To(false),
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			mutateDefaults(test.cfg)
			if diff := cmp.Diff(test.cfg, test.expected); len(diff) > 0 {
				t.Errorf("Mismatch between expected (-) and got (+):\n%s", diff)
			}
		})
	}
}
