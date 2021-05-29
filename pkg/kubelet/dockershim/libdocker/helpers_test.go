// +build !dockerless

/*
Copyright 2014 The Kubernetes Authors.

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

package libdocker

import (
	"fmt"
	"testing"

	dockertypes "github.com/docker/docker/api/types"
	"github.com/stretchr/testify/assert"
)

func TestMatchImageTagOrSHA(t *testing.T) {
	for i, testCase := range []struct {
		Inspected dockertypes.ImageInspect
		Image     string
		Output    bool
	}{
		{
			Inspected: dockertypes.ImageInspect{RepoTags: []string{"ubuntu:latest"}},
			Image:     "ubuntu",
			Output:    true,
		},
		{
			Inspected: dockertypes.ImageInspect{RepoTags: []string{"ubuntu:14.04"}},
			Image:     "ubuntu:latest",
			Output:    false,
		},
		{
			Inspected: dockertypes.ImageInspect{RepoTags: []string{"colemickens/hyperkube-amd64:217.9beff63"}},
			Image:     "colemickens/hyperkube-amd64:217.9beff63",
			Output:    true,
		},
		{
			Inspected: dockertypes.ImageInspect{RepoTags: []string{"colemickens/hyperkube-amd64:217.9beff63"}},
			Image:     "docker.io/colemickens/hyperkube-amd64:217.9beff63",
			Output:    true,
		},
		{
			Inspected: dockertypes.ImageInspect{RepoTags: []string{"docker.io/kubernetes/pause:latest"}},
			Image:     "kubernetes/pause:latest",
			Output:    true,
		},
		{
			Inspected: dockertypes.ImageInspect{
				ID: "sha256:2208f7a29005d226d1ee33a63e33af1f47af6156c740d7d23c7948e8d282d53d",
			},
			Image:  "myimage@sha256:2208f7a29005d226d1ee33a63e33af1f47af6156c740d7d23c7948e8d282d53d",
			Output: true,
		},
		{
			Inspected: dockertypes.ImageInspect{
				ID: "sha256:2208f7a29005d226d1ee33a63e33af1f47af6156c740d7d23c7948e8d282d53d",
			},
			Image:  "myimage@sha256:2208f7a29005",
			Output: false,
		},
		{
			Inspected: dockertypes.ImageInspect{
				ID: "sha256:2208f7a29005d226d1ee33a63e33af1f47af6156c740d7d23c7948e8d282d53d",
			},
			Image:  "myimage@sha256:2208",
			Output: false,
		},
		{
			// mismatched ID is ignored
			Inspected: dockertypes.ImageInspect{
				ID: "sha256:2208f7a29005d226d1ee33a63e33af1f47af6156c740d7d23c7948e8d282d53d",
			},
			Image:  "myimage@sha256:0000f7a29005d226d1ee33a63e33af1f47af6156c740d7d23c7948e8d282d53d",
			Output: false,
		},
		{
			// invalid digest is ignored
			Inspected: dockertypes.ImageInspect{
				ID: "sha256:unparseable",
			},
			Image:  "myimage@sha256:unparseable",
			Output: false,
		},
		{
			// v1 schema images can be pulled in one format and returned in another
			Inspected: dockertypes.ImageInspect{
				ID:          "sha256:9bbdf247c91345f0789c10f50a57e36a667af1189687ad1de88a6243d05a2227",
				RepoDigests: []string{"centos/ruby-23-centos7@sha256:940584acbbfb0347272112d2eb95574625c0c60b4e2fdadb139de5859cf754bf"},
			},
			Image:  "centos/ruby-23-centos7@sha256:940584acbbfb0347272112d2eb95574625c0c60b4e2fdadb139de5859cf754bf",
			Output: true,
		},
		{
			Inspected: dockertypes.ImageInspect{
				ID:       "sha256:9bbdf247c91345f0789c10f50a57e36a667af1189687ad1de88a6243d05a2227",
				RepoTags: []string{"docker.io/busybox:latest"},
			},
			Image:  "docker.io/library/busybox:latest",
			Output: true,
		},
		{
			// RepoDigest match is required
			Inspected: dockertypes.ImageInspect{
				ID:          "",
				RepoDigests: []string{"docker.io/centos/ruby-23-centos7@sha256:000084acbbfb0347272112d2eb95574625c0c60b4e2fdadb139de5859cf754bf"},
			},
			Image:  "centos/ruby-23-centos7@sha256:940584acbbfb0347272112d2eb95574625c0c60b4e2fdadb139de5859cf754bf",
			Output: false,
		},
		{
			// RepoDigest match is allowed
			Inspected: dockertypes.ImageInspect{
				ID:          "sha256:9bbdf247c91345f0789c10f50a57e36a667af1189687ad1de88a6243d05a2227",
				RepoDigests: []string{"docker.io/centos/ruby-23-centos7@sha256:940584acbbfb0347272112d2eb95574625c0c60b4e2fdadb139de5859cf754bf"},
			},
			Image:  "centos/ruby-23-centos7@sha256:940584acbbfb0347272112d2eb95574625c0c60b4e2fdadb139de5859cf754bf",
			Output: true,
		},
		{
			// RepoDigest and ID are checked
			Inspected: dockertypes.ImageInspect{
				ID:          "sha256:940584acbbfb0347272112d2eb95574625c0c60b4e2fdadb139de5859cf754bf",
				RepoDigests: []string{"docker.io/centos/ruby-23-centos7@sha256:9bbdf247c91345f0789c10f50a57e36a667af1189687ad1de88a6243d05a2227"},
			},
			Image:  "centos/ruby-23-centos7@sha256:940584acbbfb0347272112d2eb95574625c0c60b4e2fdadb139de5859cf754bf",
			Output: true,
		},
		{
			// unparseable RepoDigests are skipped
			Inspected: dockertypes.ImageInspect{
				ID: "sha256:9bbdf247c91345f0789c10f50a57e36a667af1189687ad1de88a6243d05a2227",
				RepoDigests: []string{
					"centos/ruby-23-centos7@sha256:unparseable",
					"docker.io/centos/ruby-23-centos7@sha256:940584acbbfb0347272112d2eb95574625c0c60b4e2fdadb139de5859cf754bf",
				},
			},
			Image:  "centos/ruby-23-centos7@sha256:940584acbbfb0347272112d2eb95574625c0c60b4e2fdadb139de5859cf754bf",
			Output: true,
		},
		{
			// unparseable RepoDigest is ignored
			Inspected: dockertypes.ImageInspect{
				ID:          "sha256:9bbdf247c91345f0789c10f50a57e36a667af1189687ad1de88a6243d05a2227",
				RepoDigests: []string{"docker.io/centos/ruby-23-centos7@sha256:unparseable"},
			},
			Image:  "centos/ruby-23-centos7@sha256:940584acbbfb0347272112d2eb95574625c0c60b4e2fdadb139de5859cf754bf",
			Output: false,
		},
		{
			// unparseable image digest is ignored
			Inspected: dockertypes.ImageInspect{
				ID:          "sha256:9bbdf247c91345f0789c10f50a57e36a667af1189687ad1de88a6243d05a2227",
				RepoDigests: []string{"docker.io/centos/ruby-23-centos7@sha256:unparseable"},
			},
			Image:  "centos/ruby-23-centos7@sha256:unparseable",
			Output: false,
		},
		{
			// prefix match is rejected for ID and RepoDigest
			Inspected: dockertypes.ImageInspect{
				ID:          "sha256:unparseable",
				RepoDigests: []string{"docker.io/centos/ruby-23-centos7@sha256:unparseable"},
			},
			Image:  "sha256:unparseable",
			Output: false,
		},
		{
			// possible SHA prefix match is rejected for ID and RepoDigest because it is not in the named format
			Inspected: dockertypes.ImageInspect{
				ID:          "sha256:0000f247c91345f0789c10f50a57e36a667af1189687ad1de88a6243d05a2227",
				RepoDigests: []string{"docker.io/centos/ruby-23-centos7@sha256:0000f247c91345f0789c10f50a57e36a667af1189687ad1de88a6243d05a2227"},
			},
			Image:  "sha256:0000",
			Output: false,
		},
	} {
		match := matchImageTagOrSHA(testCase.Inspected, testCase.Image)
		assert.Equal(t, testCase.Output, match, testCase.Image+fmt.Sprintf(" is not a match (%d)", i))
	}
}

func TestMatchImageIDOnly(t *testing.T) {
	for i, testCase := range []struct {
		Inspected dockertypes.ImageInspect
		Image     string
		Output    bool
	}{
		// shouldn't match names or tagged names
		{
			Inspected: dockertypes.ImageInspect{RepoTags: []string{"ubuntu:latest"}},
			Image:     "ubuntu",
			Output:    false,
		},
		{
			Inspected: dockertypes.ImageInspect{RepoTags: []string{"colemickens/hyperkube-amd64:217.9beff63"}},
			Image:     "colemickens/hyperkube-amd64:217.9beff63",
			Output:    false,
		},
		// should match name@digest refs if they refer to the image ID (but only the full ID)
		{
			Inspected: dockertypes.ImageInspect{
				ID: "sha256:2208f7a29005d226d1ee33a63e33af1f47af6156c740d7d23c7948e8d282d53d",
			},
			Image:  "myimage@sha256:2208f7a29005d226d1ee33a63e33af1f47af6156c740d7d23c7948e8d282d53d",
			Output: true,
		},
		{
			Inspected: dockertypes.ImageInspect{
				ID: "sha256:2208f7a29005d226d1ee33a63e33af1f47af6156c740d7d23c7948e8d282d53d",
			},
			Image:  "myimage@sha256:2208f7a29005",
			Output: false,
		},
		{
			Inspected: dockertypes.ImageInspect{
				ID: "sha256:2208f7a29005d226d1ee33a63e33af1f47af6156c740d7d23c7948e8d282d53d",
			},
			Image:  "myimage@sha256:2208",
			Output: false,
		},
		// should match when the IDs are literally the same
		{
			Inspected: dockertypes.ImageInspect{
				ID: "foobar",
			},
			Image:  "foobar",
			Output: true,
		},
		// shouldn't match mismatched IDs
		{
			Inspected: dockertypes.ImageInspect{
				ID: "sha256:2208f7a29005d226d1ee33a63e33af1f47af6156c740d7d23c7948e8d282d53d",
			},
			Image:  "myimage@sha256:0000f7a29005d226d1ee33a63e33af1f47af6156c740d7d23c7948e8d282d53d",
			Output: false,
		},
		// shouldn't match invalid IDs or refs
		{
			Inspected: dockertypes.ImageInspect{
				ID: "sha256:unparseable",
			},
			Image:  "myimage@sha256:unparseable",
			Output: false,
		},
		// shouldn't match against repo digests
		{
			Inspected: dockertypes.ImageInspect{
				ID:          "sha256:9bbdf247c91345f0789c10f50a57e36a667af1189687ad1de88a6243d05a2227",
				RepoDigests: []string{"centos/ruby-23-centos7@sha256:940584acbbfb0347272112d2eb95574625c0c60b4e2fdadb139de5859cf754bf"},
			},
			Image:  "centos/ruby-23-centos7@sha256:940584acbbfb0347272112d2eb95574625c0c60b4e2fdadb139de5859cf754bf",
			Output: false,
		},
	} {
		match := matchImageIDOnly(testCase.Inspected, testCase.Image)
		assert.Equal(t, testCase.Output, match, fmt.Sprintf("%s is not a match (%d)", testCase.Image, i))
	}

}
