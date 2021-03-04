/*
Copyright 2020 The Kubernetes Authors.

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

package image

import "testing"

func TestTagFromImage(t *testing.T) {
	tests := map[string]string{
		"kindest/node":         "",
		"kindest/node:latest":  "latest",
		"kindest/node:v1.17.0": "v1.17.0",
		"kindest/node:v1.17.0@sha256:9512edae126da271b66b990b6fff768fbb7cd786c7d39e86bdf55906352fdf62": "v1.17.0",
		"kindest/node@sha256:9512edae126da271b66b990b6fff768fbb7cd786c7d39e86bdf55906352fdf62":         "",

		"example.com/kindest/node":         "",
		"example.com/kindest/node:latest":  "latest",
		"example.com/kindest/node:v1.17.0": "v1.17.0",
		"example.com/kindest/node:v1.17.0@sha256:9512edae126da271b66b990b6fff768fbb7cd786c7d39e86bdf55906352fdf62": "v1.17.0",
		"example.com/kindest/node@sha256:9512edae126da271b66b990b6fff768fbb7cd786c7d39e86bdf55906352fdf62":         "",

		"example.com:3000/kindest/node":         "",
		"example.com:3000/kindest/node:latest":  "latest",
		"example.com:3000/kindest/node:v1.17.0": "v1.17.0",
		"example.com:3000/kindest/node:v1.17.0@sha256:9512edae126da271b66b990b6fff768fbb7cd786c7d39e86bdf55906352fdf62": "v1.17.0",
		"example.com:3000/kindest/node@sha256:9512edae126da271b66b990b6fff768fbb7cd786c7d39e86bdf55906352fdf62":         "",
	}
	for in, expected := range tests {
		out := TagFromImage(in)
		if out != expected {
			t.Errorf("TagFromImage(%q) = %q, expected %q instead", in, out, expected)
		}
	}
}
