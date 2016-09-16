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

package version

import "testing"

func TestParseVersion(t *testing.T) {
	cases := []struct {
		version   string
		expectErr bool
	}{
		{version: "v1.0.1-alpha"},
		{version: "v0.19.3"},
		{version: "0.19.3"},
		{version: "v1.2.0-alpha.3.1264+0655e65b435106-dirty"},
		{version: "1.2.0-alpha.3.1264+0655e65b435106-dirty"},
		{version: "1.2.0-alpha.3.1264+0655e65b435106-dirty"},
		{version: "1.0.0"},
		{version: "\t v1.0.0"},
		{version: "vv1.0.0", expectErr: true},
		{version: "blah1.0.0", expectErr: true},
	}

	for i, c := range cases {
		_, err := Parse(c.version)
		if err != nil && !c.expectErr {
			t.Errorf("[%v]unexpected error: %v", i, err)
		}
		if err == nil && c.expectErr {
			t.Errorf("[%v]expected error for %s", i, c.version)
		}
	}
}
