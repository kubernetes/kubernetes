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

func TestConfig(t *testing.T) {
	for i, tt := range []struct {
		config string
		fail   bool
	}{
		// expected failure: field "os" has numeric value, must be string
		{
			config: `
{
    "architecture": "amd64",
    "os": 123,
    "rootfs": {
      "diff_ids": [
        "sha256:5f70bf18a086007016e948b04aed3b82103a36bea41755b6cddfaf10ace3c6ef"
      ],
      "type": "layers"
    }
}
`,
			fail: true,
		},

		// expected failure: field "config.User" has numeric value, must be string
		{
			config: `
{
    "created": "2015-10-31T22:22:56.015925234Z",
    "author": "Alyssa P. Hacker <alyspdev@example.com>",
    "architecture": "amd64",
    "os": "linux",
    "config": {
        "User": 1234
    },
    "rootfs": {
      "diff_ids": [
        "sha256:5f70bf18a086007016e948b04aed3b82103a36bea41755b6cddfaf10ace3c6ef"
      ],
      "type": "layers"
    }
}
`,
			fail: true,
		},

		// expected failue: history has string value, must be an array
		{
			config: `
{
    "history": "should be an array",
    "architecture": "amd64",
    "os": 123,
    "rootfs": {
      "diff_ids": [
        "sha256:5f70bf18a086007016e948b04aed3b82103a36bea41755b6cddfaf10ace3c6ef"
      ],
      "type": "layers"
    }
}
`,
			fail: true,
		},

		// expected failure: Env has numeric value, must be a string
		{
			config: `
{
    "architecture": "amd64",
    "os": 123,
    "config": {
        "Env": [
            7353
        ]
    },
    "rootfs": {
      "diff_ids": [
        "sha256:5f70bf18a086007016e948b04aed3b82103a36bea41755b6cddfaf10ace3c6ef"
      ],
      "type": "layers"
    }
}
`,
			fail: true,
		},

		// expected failure: config.Volumes has string array, must be an object (string set)
		{
			config: `
{
    "architecture": "amd64",
    "os": 123,
    "config": {
        "Volumes": [
            "/var/job-result-data",
            "/var/log/my-app-logs"
        ]
    },
    "rootfs": {
      "diff_ids": [
        "sha256:5f70bf18a086007016e948b04aed3b82103a36bea41755b6cddfaf10ace3c6ef"
      ],
      "type": "layers"
    }
}
`,
			fail: true,
		},

		// expected failue: invalid JSON
		{
			config: `invalid JSON`,
			fail:   true,
		},

		// valid config with optional fields
		{
			config: `
{
    "created": "2015-10-31T22:22:56.015925234Z",
    "author": "Alyssa P. Hacker <alyspdev@example.com>",
    "architecture": "amd64",
    "os": "linux",
    "config": {
        "User": "1:1",
        "ExposedPorts": {
            "8080/tcp": {}
        },
        "Env": [
            "PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
            "FOO=docker_is_a_really",
            "BAR=great_tool_you_know"
        ],
        "Entrypoint": [
            "/bin/sh"
        ],
        "Cmd": [
            "--foreground",
            "--config",
            "/etc/my-app.d/default.cfg"
        ],
        "Volumes": {
            "/var/job-result-data": {},
            "/var/log/my-app-logs": {}
        },
        "StopSignal": "SIGKILL",
        "WorkingDir": "/home/alice",
        "Labels": {
            "com.example.project.git.url": "https://example.com/project.git",
            "com.example.project.git.commit": "45a939b2999782a3f005621a8d0f29aa387e1d6b"
        }
    },
    "rootfs": {
      "diff_ids": [
        "sha256:9d3dd9504c685a304985025df4ed0283e47ac9ffa9bd0326fddf4d59513f0827",
        "sha256:2b689805fbd00b2db1df73fae47562faac1a626d5f61744bfe29946ecff5d73d"
      ],
      "type": "layers"
    },
    "history": [
      {
        "created": "2015-10-31T22:22:54.690851953Z",
        "created_by": "/bin/sh -c #(nop) ADD file:a3bc1e842b69636f9df5256c49c5374fb4eef1e281fe3f282c65fb853ee171c5 in /"
      },
      {
        "created": "2015-10-31T22:22:55.613815829Z",
        "created_by": "/bin/sh -c #(nop) CMD [\"sh\"]",
        "empty_layer": true
      }
    ]
}
`,
			fail: false,
		},

		// valid config with only required fields
		{
			config: `
{
    "architecture": "amd64",
    "os": "linux",
    "rootfs": {
      "diff_ids": [
        "sha256:5f70bf18a086007016e948b04aed3b82103a36bea41755b6cddfaf10ace3c6ef"
      ],
      "type": "layers"
    }
}
`,
			fail: false,
		},
	} {
		r := strings.NewReader(tt.config)
		err := schema.ValidatorMediaTypeImageConfig.Validate(r)

		if got := err != nil; tt.fail != got {
			t.Errorf("test %d: expected validation failure %t but got %t, err %v", i, tt.fail, got, err)
		}
	}
}
