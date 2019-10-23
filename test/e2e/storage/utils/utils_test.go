/*
Copyright 2019 The Kubernetes Authors.

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

package utils

import (
	"testing"

	"github.com/onsi/gomega"
)

func TestFindGlobalVolumeMountPaths(t *testing.T) {
	tests := []struct {
		name      string
		mountInfo string
		podUID    string
		expected  []string
	}{
		{
			name: "pod uses local filesystem pv with block source",
			mountInfo: `1045 245 0:385 / /var/lib/kubelet/pods/ff5e9fa2-7111-486d-854c-848bcc6b3819/volumes/kubernetes.io~secret/default-token-djlt2 rw,relatime shared:199 - tmpfs tmpfs rw
1047 245 7:6 / /var/lib/kubelet/plugins/kubernetes.io/local-volume/mounts/local-wdx8b rw,relatime shared:200 - ext4 /dev/loop6 rw,data=ordered
1048 245 7:6 / /var/lib/kubelet/pods/ff5e9fa2-7111-486d-854c-848bcc6b3819/volumes/kubernetes.io~local-volume/local-wdx8b rw,relatime shared:200 - ext4 /dev/loop6 rw,data=ordered
1054 245 7:6 /provisioning-9823 /var/lib/kubelet/pods/ff5e9fa2-7111-486d-854c-848bcc6b3819/volume-subpaths/local-wdx8b/test-container-subpath-local-preprovisionedpv-d72p/0 rw,relatime shared:200 - ext4 /dev/loop6 rw,data=ordered
`,
			podUID: "ff5e9fa2-7111-486d-854c-848bcc6b3819",
			expected: []string{
				"/var/lib/kubelet/plugins/kubernetes.io/local-volume/mounts/local-wdx8b",
			},
		},
	}

	g := gomega.NewWithT(t)
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			mountPaths, err := findGlobalVolumeMountPaths(tt.mountInfo, tt.podUID)
			if err != nil {
				t.Fatal(err)
			}
			g.Expect(mountPaths).To(gomega.ConsistOf(tt.expected))
		})
	}
}
