/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package validation

import (
	"strings"
	"testing"
)

func TestValidatePathSegmentName(t *testing.T) {
	testcases := map[string]struct {
		Name        string
		Prefix      bool
		ExpectedOK  bool
		ExpectedMsg string
	}{
		"empty": {
			Name:        "",
			Prefix:      false,
			ExpectedOK:  true,
			ExpectedMsg: "",
		},
		"empty,prefix": {
			Name:        "",
			Prefix:      true,
			ExpectedOK:  true,
			ExpectedMsg: "",
		},

		"valid": {
			Name:        "foo.bar.baz",
			Prefix:      false,
			ExpectedOK:  true,
			ExpectedMsg: "",
		},
		"valid,prefix": {
			Name:        "foo.bar.baz",
			Prefix:      true,
			ExpectedOK:  true,
			ExpectedMsg: "",
		},

		// Make sure mixed case, non DNS subdomain characters are tolerated
		"valid complex": {
			Name:        "sha256:ABCDEF012345@ABCDEF012345",
			Prefix:      false,
			ExpectedOK:  true,
			ExpectedMsg: "",
		},
		// Make sure non-ascii characters are tolerated
		"valid extended charset": {
			Name:        "Iñtërnâtiônàlizætiøn",
			Prefix:      false,
			ExpectedOK:  true,
			ExpectedMsg: "",
		},

		"dot": {
			Name:        ".",
			Prefix:      false,
			ExpectedOK:  false,
			ExpectedMsg: ".",
		},
		"dot leading": {
			Name:        ".test",
			Prefix:      false,
			ExpectedOK:  true,
			ExpectedMsg: "",
		},
		"dot,prefix": {
			Name:        ".",
			Prefix:      true,
			ExpectedOK:  true, // allowed because a suffix could make it valid
			ExpectedMsg: "",
		},

		"dot dot": {
			Name:        "..",
			Prefix:      false,
			ExpectedOK:  false,
			ExpectedMsg: "..",
		},
		"dot dot leading": {
			Name:        "..test",
			Prefix:      false,
			ExpectedOK:  true,
			ExpectedMsg: "",
		},
		"dot dot,prefix": {
			Name:        "..",
			Prefix:      true,
			ExpectedOK:  true, // allowed because a suffix could make it valid
			ExpectedMsg: "",
		},

		"slash": {
			Name:        "foo/bar",
			Prefix:      false,
			ExpectedOK:  false,
			ExpectedMsg: "/",
		},
		"slash,prefix": {
			Name:        "foo/bar",
			Prefix:      true,
			ExpectedOK:  false,
			ExpectedMsg: "/",
		},

		"percent": {
			Name:        "foo%bar",
			Prefix:      false,
			ExpectedOK:  false,
			ExpectedMsg: "%",
		},
		"percent,prefix": {
			Name:        "foo%bar",
			Prefix:      true,
			ExpectedOK:  false,
			ExpectedMsg: "%",
		},
	}

	for k, tc := range testcases {
		ok, msg := ValidatePathSegmentName(tc.Name, tc.Prefix)
		if ok != tc.ExpectedOK {
			t.Errorf("%s: expected ok=%v, got %v", k, tc.ExpectedOK, ok)
		}
		if len(tc.ExpectedMsg) == 0 && len(msg) > 0 {
			t.Errorf("%s: expected no message, got %v", k, msg)
		}
		if len(tc.ExpectedMsg) > 0 && !strings.Contains(msg, tc.ExpectedMsg) {
			t.Errorf("%s: expected message containing %q, got %v", k, tc.ExpectedMsg, msg)
		}
	}
}
