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

package conntrack

import (
	"k8s.io/ktesting"
	kubeproxyconfig "k8s.io/kubernetes/pkg/proxy/apis/config"
	"k8s.io/utils/ptr"
	"testing"
)

func TestGetConntrackMaxConfiguredCap(t *testing.T) {
	for _, tc := range []struct {
		name    string
		cores   int
		perCore int32
		floor   int32
		cap     *int32
		want    int
	}{
		{"default cap", 512, 32768, 131072, nil, 1048576},
		{"raised cap", 1024, 32768, 131072, ptr.To[int32](22000000), 22000000},
		{"below raised cap", 128, 32768, 131072, ptr.To[int32](22000000), 4194304},
		{"lower cap", 128, 32768, 131072, ptr.To[int32](524288), 524288},
		{"fixed limit on small host", 2, 32768, 22000000, ptr.To[int32](22000000), 22000000},
		{"fixed limit on large host", 1024, 32768, 22000000, ptr.To[int32](22000000), 22000000},
		{"uncapped", 512, 32768, 131072, ptr.To[int32](0), 16777216},
		{"preserve host limit", 512, 0, 22000000, ptr.To[int32](22000000), 0},
	} {
		t.Run(tc.name, func(t *testing.T) {
			_, ctx := ktesting.NewTestContext(t)
			got, err := getConntrackMax(ctx, &kubeproxyconfig.KubeProxyConntrackConfiguration{MaxPerCore: ptr.To(tc.perCore), Min: ptr.To(tc.floor), Max: tc.cap}, tc.cores)
			if err != nil || got != tc.want {
				t.Fatalf("got (%d, %v), want (%d, nil)", got, err, tc.want)
			}
		})
	}
}
