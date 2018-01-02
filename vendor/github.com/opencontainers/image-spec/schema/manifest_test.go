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

func TestManifest(t *testing.T) {
	for i, tt := range []struct {
		manifest string
		fail     bool
	}{
		// expected failure: mediaType does not match pattern
		{
			manifest: `
{
  "schemaVersion": 2,
  "config": {
    "mediaType": "invalid",
    "size": 1470,
    "digest": "sha256:c86f7763873b6c0aae22d963bab59b4f5debbed6685761b5951584f6efb0633b"
  },
  "layers": [
    {
      "mediaType": "application/vnd.oci.image.layer.v1.tar+gzip",
      "size": 148,
      "digest": "sha256:c57089565e894899735d458f0fd4bb17a0f1e0df8d72da392b85c9b35ee777cd"
    }
  ]
}
`,
			fail: true,
		},

		// expected failure: config.size is a string, expected integer
		{
			manifest: `
{
  "schemaVersion": 2,
  "config": {
    "mediaType": "application/vnd.oci.image.config.v1+json",
    "size": "1470",
    "digest": "sha256:c86f7763873b6c0aae22d963bab59b4f5debbed6685761b5951584f6efb0633b"
  },
  "layers": [
    {
      "mediaType": "application/vnd.oci.image.layer.v1.tar+gzip",
      "size": 148,
      "digest": "sha256:c57089565e894899735d458f0fd4bb17a0f1e0df8d72da392b85c9b35ee777cd"
    }
  ]
}
`,
			fail: true,
		},

		// expected failure: layers.size is string, expected integer
		{
			manifest: `
{
  "schemaVersion": 2,
  "config": {
    "mediaType": "application/vnd.oci.image.config.v1+json",
    "size": 1470,
    "digest": "sha256:c86f7763873b6c0aae22d963bab59b4f5debbed6685761b5951584f6efb0633b"
  },
  "layers": [
    {
      "mediaType": "application/vnd.oci.image.layer.v1.tar+gzip",
      "size": "675598",
      "digest": "sha256:c86f7763873b6c0aae22d963bab59b4f5debbed6685761b5951584f6efb0633b"
    }
  ]
}
`,
			fail: true,
		},

		// valid manifest with optional fields
		{
			manifest: `
{
  "schemaVersion": 2,
  "config": {
    "mediaType": "application/vnd.oci.image.config.v1+json",
    "size": 1470,
    "digest": "sha256:c86f7763873b6c0aae22d963bab59b4f5debbed6685761b5951584f6efb0633b"
  },
  "layers": [
    {
      "mediaType": "application/vnd.oci.image.layer.v1.tar+gzip",
      "size": 675598,
      "digest": "sha256:9d3dd9504c685a304985025df4ed0283e47ac9ffa9bd0326fddf4d59513f0827"
    },
    {
      "mediaType": "application/vnd.oci.image.layer.v1.tar+gzip",
      "size": 156,
      "digest": "sha256:2b689805fbd00b2db1df73fae47562faac1a626d5f61744bfe29946ecff5d73d"
    },
    {
      "mediaType": "application/vnd.oci.image.layer.v1.tar+gzip",
      "size": 148,
      "digest": "sha256:c57089565e894899735d458f0fd4bb17a0f1e0df8d72da392b85c9b35ee777cd"
    }
  ],
  "annotations": {
    "key1": "value1",
    "key2": "value2"
  }
}
`,
			fail: false,
		},

		// valid manifest with only required fields
		{
			manifest: `
{
  "schemaVersion": 2,
  "config": {
    "mediaType": "application/vnd.oci.image.config.v1+json",
    "size": 1470,
    "digest": "sha256:c86f7763873b6c0aae22d963bab59b4f5debbed6685761b5951584f6efb0633b"
  },
  "layers": [
    {
      "mediaType": "application/vnd.oci.image.layer.v1.tar+gzip",
      "size": 675598,
      "digest": "sha256:9d3dd9504c685a304985025df4ed0283e47ac9ffa9bd0326fddf4d59513f0827"
    },
    {
      "mediaType": "application/vnd.oci.image.layer.v1.tar+gzip",
      "size": 156,
      "digest": "sha256:2b689805fbd00b2db1df73fae47562faac1a626d5f61744bfe29946ecff5d73d"
    },
    {
      "mediaType": "application/vnd.oci.image.layer.v1.tar+gzip",
      "size": 148,
      "digest": "sha256:c57089565e894899735d458f0fd4bb17a0f1e0df8d72da392b85c9b35ee777cd"
    }
  ]
}
`,
			fail: false,
		},

		// expected failure: empty layer, expected at least one
		{
			manifest: `
{
  "schemaVersion": 2,
  "config": {
    "mediaType": "application/vnd.oci.image.config.v1+json",
    "size": 1470,
    "digest": "sha256:c86f7763873b6c0aae22d963bab59b4f5debbed6685761b5951584f6efb0633b"
  },
  "layers": []
}
`,
			fail: true,
		},

		// expected pass: test bounds of algorithm field in digest.
		{
			manifest: `
{
  "schemaVersion": 2,
  "config": {
    "mediaType": "application/vnd.oci.image.config.v1+json",
    "size": 1470,
    "digest": "sha256+b64:c86f7763873b6c0aae22d963bab59b4f5debbed6685761b5951584f6efb0633b"
  },
  "layers": [
    {
      "mediaType": "application/vnd.oci.image.config.v1+json",
      "size": 1470,
      "digest": "sha256+foo-bar:c86f7763873b6c0aae22d963bab59b4f5debbed6685761b5951584f6efb0633b"
    },
    {
      "mediaType": "application/vnd.oci.image.config.v1+json",
      "size": 1470,
      "digest": "sha256.foo-bar:c86f7763873b6c0aae22d963bab59b4f5debbed6685761b5951584f6efb0633b"
    },
    {
      "mediaType": "application/vnd.oci.image.config.v1+json",
      "size": 1470,
	  "digest": "multihash+base58:QmRZxt2b1FVZPNqd8hsiykDL3TdBDeTSPX9Kv46HmX4Gx8"
    }
  ]
}
`,
		},

		// expected failure: push bounds of algorithm field in digest too far.
		{
			manifest: `
{
  "schemaVersion": 2,
  "config": {
    "mediaType": "application/vnd.oci.image.config.v1+json",
    "size": 1470,
    "digest": "sha256+b64:c86f7763873b6c0aae22d963bab59b4f5debbed6685761b5951584f6efb0633b"
  },
  "layers": [
    {
      "mediaType": "application/vnd.oci.image.config.v1+json",
      "size": 1470,
      "digest": "sha256+foo+-b:c86f7763873b6c0aae22d963bab59b4f5debbed6685761b5951584f6efb0633b"
    }
  ]
}
`,
			fail: true,
		},
	} {
		r := strings.NewReader(tt.manifest)
		err := schema.ValidatorMediaTypeManifest.Validate(r)

		if got := err != nil; tt.fail != got {
			t.Errorf("test %d: expected validation failure %t but got %t, err %v", i, tt.fail, got, err)
		}
	}
}
