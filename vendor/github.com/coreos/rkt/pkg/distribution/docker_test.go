// Copyright 2016 The rkt Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package distribution

import (
	"net/url"
	"testing"
)

func TestDocker(t *testing.T) {
	tests := []struct {
		dockerRef string

		expectedCIMD string
		expected     string
	}{
		{
			"busybox",
			"cimd:docker:v=0:registry-1.docker.io/library/busybox:latest",
			"busybox",
		},
		{
			"busybox:latest",
			"cimd:docker:v=0:registry-1.docker.io/library/busybox:latest",
			"busybox",
		},
		{
			"registry-1.docker.io/library/busybox:latest",
			"cimd:docker:v=0:registry-1.docker.io/library/busybox:latest",
			"busybox",
		},
		{
			"busybox:1.0",
			"cimd:docker:v=0:registry-1.docker.io/library/busybox:1.0",
			"busybox:1.0",
		},
		{
			"repo/image",
			"cimd:docker:v=0:registry-1.docker.io/repo/image:latest",
			"repo/image",
		},
		{
			"repo/image:latest",
			"cimd:docker:v=0:registry-1.docker.io/repo/image:latest",
			"repo/image",
		},
		{
			"repo/image:1.0",
			"cimd:docker:v=0:registry-1.docker.io/repo/image:1.0",
			"repo/image:1.0",
		},
		{
			"busybox@sha256:a59906e33509d14c036c8678d687bd4eec81ed7c4b8ce907b888c607f6a1e0e6",
			"cimd:docker:v=0:registry-1.docker.io/library/busybox@sha256:a59906e33509d14c036c8678d687bd4eec81ed7c4b8ce907b888c607f6a1e0e6",
			"busybox@sha256:a59906e33509d14c036c8678d687bd4eec81ed7c4b8ce907b888c607f6a1e0e6",
		},
		{
			"myregistry.example.com:4000/busybox",
			"cimd:docker:v=0:myregistry.example.com:4000/busybox:latest",
			"myregistry.example.com:4000/busybox",
		},
		{
			"myregistry.example.com:4000/busybox:latest",
			"cimd:docker:v=0:myregistry.example.com:4000/busybox:latest",
			"myregistry.example.com:4000/busybox",
		},
		{
			"myregistry.example.com:4000/busybox:1.0",
			"cimd:docker:v=0:myregistry.example.com:4000/busybox:1.0",
			"myregistry.example.com:4000/busybox:1.0",
		},
	}

	for _, tt := range tests {
		// Test NewDockerFromDockerString
		d, err := NewDockerFromString(tt.dockerRef)
		if err != nil {
			t.Errorf("unexpected error: %v", err)
			continue
		}

		u, err := url.Parse(tt.expectedCIMD)
		if err != nil {
			t.Errorf("unexpected error: %v", err)
			continue
		}

		td, err := NewDocker(u)
		if err != nil {
			t.Errorf("unexpected error: %v", err)
			continue
		}

		if !d.Equals(td) {
			t.Errorf("expected identical distribution but got %q != %q", td.CIMD().String(), d.CIMD().String())
		}
	}
}
