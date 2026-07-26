//go:build linux

/*
Copyright 2026 The Kubernetes Authors.

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

package kubelet

import "testing"

func TestIsNativeLogger(t *testing.T) {
	tests := []struct {
		name    string
		output  string
		service string
		want    bool
	}{
		{
			name:    "matches an exact unit",
			output:  "containerd.service\nkubelet.service\n",
			service: "kubelet",
			want:    true,
		},
		{
			name:    "does not match a unit containing the requested name",
			output:  "mykubelet.service\n",
			service: "kubelet",
			want:    false,
		},
		{
			name:    "does not match a unit with the requested name as a prefix",
			output:  "kubelet.service.backup\n",
			service: "kubelet",
			want:    false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := isNativeLogger([]byte(tt.output), tt.service); got != tt.want {
				t.Errorf("isNativeLogger(%q, %q) = %t, want %t", tt.output, tt.service, got, tt.want)
			}
		})
	}
}
