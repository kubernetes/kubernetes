/*
Copyright (c) 2016 VMware, Inc. All Rights Reserved.

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

package object

import "testing"

func TestParseDatastorePath(t *testing.T) {
	tests := []struct {
		dsPath string
		dsFile string
		fail   bool
	}{
		{"", "", true},
		{"x", "", true},
		{"[", "", true},
		{"[nope", "", true},
		{"[te st]", "", false},
		{"[te st] foo", "foo", false},
		{"[te st] foo/foo.vmx", "foo/foo.vmx", false},
		{"[te st]foo bar/foo bar.vmx", "foo bar/foo bar.vmx", false},
		{" [te st]     bar/bar.vmx  ", "bar/bar.vmx", false},
	}

	for _, test := range tests {
		p := new(DatastorePath)
		ok := p.FromString(test.dsPath)

		if test.fail {
			if ok {
				t.Errorf("expected error for: %s", test.dsPath)
			}
		} else {
			if !ok {
				t.Errorf("failed to parse: %q", test.dsPath)
			} else {
				if test.dsFile != p.Path {
					t.Errorf("dsFile=%s", p.Path)
				}
				if p.Datastore != "te st" {
					t.Errorf("ds=%s", p.Datastore)
				}
			}
		}
	}

	s := "[datastore1] foo/bar.vmdk"
	p := new(DatastorePath)
	ok := p.FromString(s)
	if !ok {
		t.Fatal(s)
	}

	if p.String() != s {
		t.Fatal(p.String())
	}

	p.Path = ""

	if p.String() != "[datastore1]" {
		t.Fatal(p.String())
	}
}
