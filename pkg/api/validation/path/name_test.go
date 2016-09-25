/*
Copyright 2015 The Kubernetes Authors.

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

package path

import (
	"strings"
	"testing"
)

func TestValidatePathSegmentName(t *testing.T) {
	testcases := map[string]struct {
		Name        string
		Prefix      bool
		ExpectedMsg string
	}{
		"empty": {
			Name:        "",
			Prefix:      false,
			ExpectedMsg: "",
		},
		"empty,prefix": {
			Name:        "",
			Prefix:      true,
			ExpectedMsg: "",
		},

		"valid": {
			Name:        "foo.bar.baz",
			Prefix:      false,
			ExpectedMsg: "",
		},
		"valid,prefix": {
			Name:        "foo.bar.baz",
			Prefix:      true,
			ExpectedMsg: "",
		},

		// Make sure mixed case, non DNS subdomain characters are tolerated
		"valid complex": {
			Name:        "sha256:ABCDEF012345@ABCDEF012345",
			Prefix:      false,
			ExpectedMsg: "",
		},
		// Make sure non-ascii characters are tolerated
		"valid extended charset": {
			Name:        "Iñtërnâtiônàlizætiøn",
			Prefix:      false,
			ExpectedMsg: "",
		},

		"dot": {
			Name:        ".",
			Prefix:      false,
			ExpectedMsg: ".",
		},
		"dot leading": {
			Name:        ".test",
			Prefix:      false,
			ExpectedMsg: "",
		},
		"dot,prefix": {
			Name:        ".",
			Prefix:      true,
			ExpectedMsg: "",
		},

		"dot dot": {
			Name:        "..",
			Prefix:      false,
			ExpectedMsg: "..",
		},
		"dot dot leading": {
			Name:        "..test",
			Prefix:      false,
			ExpectedMsg: "",
		},
		"dot dot,prefix": {
			Name:        "..",
			Prefix:      true,
			ExpectedMsg: "",
		},

		"slash": {
			Name:        "foo/bar",
			Prefix:      false,
			ExpectedMsg: "/",
		},
		"slash,prefix": {
			Name:        "foo/bar",
			Prefix:      true,
			ExpectedMsg: "/",
		},

		"percent": {
			Name:        "foo%bar",
			Prefix:      false,
			ExpectedMsg: "%",
		},
		"percent,prefix": {
			Name:        "foo%bar",
			Prefix:      true,
			ExpectedMsg: "%",
		},
	}

	for k, tc := range testcases {
		msgs := ValidatePathSegmentName(tc.Name, tc.Prefix)
		if len(tc.ExpectedMsg) == 0 && len(msgs) > 0 {
			t.Errorf("%s: expected no message, got %v", k, msgs)
		}
		if len(tc.ExpectedMsg) > 0 && len(msgs) == 0 {
			t.Errorf("%s: expected error message, got none", k)
		}
		if len(tc.ExpectedMsg) > 0 && !strings.Contains(msgs[0], tc.ExpectedMsg) {
			t.Errorf("%s: expected message containing %q, got %v", k, tc.ExpectedMsg, msgs[0])
		}
	}
}
