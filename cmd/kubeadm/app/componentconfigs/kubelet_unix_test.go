//go:build !windows
// +build !windows

/*
Copyright 2024 The Kubernetes Authors.

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
	"reflect"
	"testing"

	kubeletconfig "k8s.io/kubelet/config/v1beta1"
	"k8s.io/utils/ptr"
)

func TestMutateResolverConfig(t *testing.T) {
	var fooResolverConfig = "/foo/resolver"

	tests := []struct {
		name                string
		cfg                 *kubeletconfig.KubeletConfiguration
		isServiceActiveFunc func(string) (bool, error)
		expected            *kubeletconfig.KubeletConfiguration
	}{
		{
			name: "the resolver config should not be mutated when it was set already even if systemd-resolved is active",
			cfg: &kubeletconfig.KubeletConfiguration{
				ResolverConfig: ptr.To(fooResolverConfig),
			},
			isServiceActiveFunc: func(string) (bool, error) { return true, nil },
			expected: &kubeletconfig.KubeletConfiguration{
				ResolverConfig: ptr.To(fooResolverConfig),
			},
		},
		{
			name: "the resolver config should be set when systemd-resolved is active",
			cfg: &kubeletconfig.KubeletConfiguration{
				ResolverConfig: nil,
			},
			isServiceActiveFunc: func(string) (bool, error) { return true, nil },
			expected: &kubeletconfig.KubeletConfiguration{
				ResolverConfig: ptr.To(kubeletSystemdResolverConfig),
			},
		},
		{
			name: "the resolver config should not be set when systemd-resolved is not active",
			cfg: &kubeletconfig.KubeletConfiguration{
				ResolverConfig: nil,
			},
			isServiceActiveFunc: func(string) (bool, error) { return false, nil },
			expected: &kubeletconfig.KubeletConfiguration{
				ResolverConfig: nil,
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			err := mutateResolverConfig(test.cfg, test.isServiceActiveFunc)
			if err != nil {
				t.Fatalf("failed to mutate ResolverConfig for KubeletConfiguration, %v", err)
			}
			if !reflect.DeepEqual(test.cfg, test.expected) {
				t.Errorf("Mismatch between expected and got:\nExpected:\n%+v\n---\nGot:\n%+v",
					test.expected, test.cfg)
			}
		})
	}
}
