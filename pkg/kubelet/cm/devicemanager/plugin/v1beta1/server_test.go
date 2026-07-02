/*
Copyright The Kubernetes Authors.

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

package v1beta1

import (
	"context"
	"strings"
	"testing"

	api "k8s.io/kubelet/pkg/apis/deviceplugin/v1beta1"
)

func TestRegisterRejectsEndpointOutsideSocketDir(t *testing.T) {
	const socketDir = "/var/lib/kubelet/device-plugins/"

	for _, tc := range []struct {
		name     string
		endpoint string
	}{
		{"parent traversal", "../../../../tmp/evil.sock"},
		{"traversal back into dir then out", "../device-plugins/../../tmp/evil.sock"},
		{"subdirectory", "sub/dir.sock"},
		{"absolute path", "/tmp/evil.sock"},
		{"empty", ""},
		{"dot", "."},
		{"dotdot", ".."},
	} {
		t.Run(tc.name, func(t *testing.T) {
			s := &server{
				socketDir: socketDir,
				clients:   make(map[string]Client),
			}
			_, err := s.Register(context.Background(), &api.RegisterRequest{
				Version:      api.Version,
				Endpoint:     tc.endpoint,
				ResourceName: "example.com/device",
			})
			if err == nil {
				t.Fatalf("endpoint %q: expected an error, got nil", tc.endpoint)
			}
			const want = "must name a socket directly within the device plugin directory"
			if !strings.Contains(err.Error(), want) {
				t.Fatalf("endpoint %q: expected invalid-endpoint error, got: %v", tc.endpoint, err)
			}
			if len(s.clients) != 0 {
				t.Fatalf("endpoint %q: connectClient was reached, clients=%v", tc.endpoint, s.clients)
			}
		})
	}
}
