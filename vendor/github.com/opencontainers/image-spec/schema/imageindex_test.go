// Copyright 2016 The Linux Foundation
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

package schema_test

import (
	"strings"
	"testing"

	"github.com/opencontainers/image-spec/schema"
)

func TestImageIndex(t *testing.T) {
	for i, tt := range []struct {
		imageIndex string
		fail       bool
	}{
		// expected failure: mediaType does not match pattern
		{
			imageIndex: `
{
  "schemaVersion": 2,
  "manifests": [
    {
      "mediaType": "invalid",
      "size": 7143,
      "digest": "sha256:e692418e4cbaf90ca69d05a66403747baa33ee08806650b51fab815ad7fc331f",
      "platform": {
        "architecture": "ppc64le",
        "os": "linux"
      }
    }
  ]
}
`,
			fail: true,
		},

		// expected failure: manifest.size is string, expected integer
		{
			imageIndex: `
{
  "schemaVersion": 2,
  "manifests": [
    {
      "mediaType": "application/vnd.oci.image.manifest.v1+json",
      "size": "7682",
      "digest": "sha256:5b0bcabd1ed22e9fb1310cf6c2dec7cdef19f0ad69efa1f392e94a4333501270",
      "platform": {
        "architecture": "amd64",
        "os": "linux"
      }
    }
  ]
}
`,
			fail: true,
		},

		// expected failure: manifest.digest is missing, expected required
		{
			imageIndex: `
{
  "schemaVersion": 2,
  "manifests": [
    {
      "mediaType": "application/vnd.oci.image.manifest.v1+json",
      "size": 7682,
      "platform": {
        "architecture": "amd64",
        "os": "linux"
      }
    }
  ]
}
`,
			fail: true,
		},

		// expected failure: in the optional field platform platform.architecture is missing, expected required
		{
			imageIndex: `
{
  "schemaVersion": 2,
  "manifests": [
    {
      "mediaType": "application/vnd.oci.image.manifest.v1+json",
      "size": 7682,
      "digest": "sha256:5b0bcabd1ed22e9fb1310cf6c2dec7cdef19f0ad69efa1f392e94a4333501270",
      "platform": {
	"os": "linux",
      }
    }
  ]
}
`,
			fail: true,
		},

		// expected failure: invalid referenced manifest media type
		{
			imageIndex: `
{
  "schemaVersion": 2,
  "manifests": [
    {
      "mediaType": "invalid",
      "size": 7682,
      "digest": "sha256:5b0bcabd1ed22e9fb1310cf6c2dec7cdef19f0ad69efa1f392e94a4333501270",
      "platform": {
        "architecture": "amd64",
        "os": "linux"
      }
    }
  ]
}
`,
			fail: true,
		},

		// expected failure: empty referenced manifest media type
		{
			imageIndex: `
{
  "schemaVersion": 2,
  "manifests": [
    {
      "mediaType": "",
      "size": 7682,
      "digest": "sha256:5b0bcabd1ed22e9fb1310cf6c2dec7cdef19f0ad69efa1f392e94a4333501270",
      "platform": {
        "architecture": "amd64",
        "os": "linux"
      }
    }
  ]
}
`,
			fail: true,
		},

		// valid image index, with optional fields
		{
			imageIndex: `
{
  "schemaVersion": 2,
  "manifests": [
    {
      "mediaType": "application/vnd.oci.image.manifest.v1+json",
      "size": 7143,
      "digest": "sha256:e692418e4cbaf90ca69d05a66403747baa33ee08806650b51fab815ad7fc331f",
      "platform": {
        "architecture": "ppc64le",
        "os": "linux"
      }
    },
    {
      "mediaType": "application/vnd.oci.image.manifest.v1+json",
      "size": 7682,
      "digest": "sha256:5b0bcabd1ed22e9fb1310cf6c2dec7cdef19f0ad69efa1f392e94a4333501270",
      "platform": {
        "architecture": "amd64",
        "os": "linux"
      }
    }
  ],
  "annotations": {
    "com.example.key1": "value1",
    "com.example.key2": "value2"
  }
}
`,
			fail: false,
		},

		// valid image index, with required fields only
		{
			imageIndex: `
{
  "schemaVersion": 2,
  "manifests": [
    {
      "mediaType": "application/vnd.oci.image.manifest.v1+json",
      "size": 7143,
      "digest": "sha256:e692418e4cbaf90ca69d05a66403747baa33ee08806650b51fab815ad7fc331f"
    }
  ]
}
`,
			fail: false,
		},

		// valid image index, with customized media type of referenced manifest
		{
			imageIndex: `
{
  "schemaVersion": 2,
  "manifests": [
    {
      "mediaType": "application/customized.manifest+json",
      "size": 7143,
      "digest": "sha256:e692418e4cbaf90ca69d05a66403747baa33ee08806650b51fab815ad7fc331f",
      "platform": {
        "architecture": "ppc64le",
        "os": "linux"
      }
    }
  ]
}
`,
			fail: false,
		},
	} {
		r := strings.NewReader(tt.imageIndex)
		err := schema.ValidatorMediaTypeImageIndex.Validate(r)

		if got := err != nil; tt.fail != got {
			t.Errorf("test %d: expected validation failure %t but got %t, err %v", i, tt.fail, got, err)
		}
	}
}
