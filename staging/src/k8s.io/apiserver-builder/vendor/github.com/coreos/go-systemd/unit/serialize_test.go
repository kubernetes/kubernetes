// Copyright 2015 CoreOS, Inc.
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

package unit

import (
	"io/ioutil"
	"testing"
)

func TestSerialize(t *testing.T) {
	tests := []struct {
		input  []*UnitOption
		output string
	}{
		// no options results in empty file
		{
			[]*UnitOption{},
			``,
		},

		// options with same section share the header
		{
			[]*UnitOption{
				&UnitOption{"Unit", "Description", "Foo"},
				&UnitOption{"Unit", "BindsTo", "bar.service"},
			},
			`[Unit]
Description=Foo
BindsTo=bar.service
`,
		},

		// options with same name are not combined
		{
			[]*UnitOption{
				&UnitOption{"Unit", "Description", "Foo"},
				&UnitOption{"Unit", "Description", "Bar"},
			},
			`[Unit]
Description=Foo
Description=Bar
`,
		},

		// multiple options printed under different section headers
		{
			[]*UnitOption{
				&UnitOption{"Unit", "Description", "Foo"},
				&UnitOption{"Service", "ExecStart", "/usr/bin/sleep infinity"},
			},
			`[Unit]
Description=Foo

[Service]
ExecStart=/usr/bin/sleep infinity
`,
		},

		// options are grouped into sections
		{
			[]*UnitOption{
				&UnitOption{"Unit", "Description", "Foo"},
				&UnitOption{"Service", "ExecStart", "/usr/bin/sleep infinity"},
				&UnitOption{"Unit", "BindsTo", "bar.service"},
			},
			`[Unit]
Description=Foo
BindsTo=bar.service

[Service]
ExecStart=/usr/bin/sleep infinity
`,
		},

		// options are ordered within groups, and sections are ordered in the order in which they were first seen
		{
			[]*UnitOption{
				&UnitOption{"Unit", "Description", "Foo"},
				&UnitOption{"Service", "ExecStart", "/usr/bin/sleep infinity"},
				&UnitOption{"Unit", "BindsTo", "bar.service"},
				&UnitOption{"X-Foo", "Bar", "baz"},
				&UnitOption{"Service", "ExecStop", "/usr/bin/sleep 1"},
				&UnitOption{"Unit", "Documentation", "https://foo.com"},
			},
			`[Unit]
Description=Foo
BindsTo=bar.service
Documentation=https://foo.com

[Service]
ExecStart=/usr/bin/sleep infinity
ExecStop=/usr/bin/sleep 1

[X-Foo]
Bar=baz
`,
		},

		// utf8 characters are not a problem
		{
			[]*UnitOption{
				&UnitOption{"©", "µ☃", "ÇôrèÕ$"},
			},
			`[©]
µ☃=ÇôrèÕ$
`,
		},

		// no verification is done on section names
		{
			[]*UnitOption{
				&UnitOption{"Un\nit", "Description", "Foo"},
			},
			`[Un
it]
Description=Foo
`,
		},

		// no verification is done on option names
		{
			[]*UnitOption{
				&UnitOption{"Unit", "Desc\nription", "Foo"},
			},
			`[Unit]
Desc
ription=Foo
`,
		},

		// no verification is done on option values
		{
			[]*UnitOption{
				&UnitOption{"Unit", "Description", "Fo\no"},
			},
			`[Unit]
Description=Fo
o
`,
		},
	}

	for i, tt := range tests {
		outReader := Serialize(tt.input)
		outBytes, err := ioutil.ReadAll(outReader)
		if err != nil {
			t.Errorf("case %d: encountered error while reading output: %v", i, err)
			continue
		}

		output := string(outBytes)
		if tt.output != output {
			t.Errorf("case %d: incorrect output", i)
			t.Logf("Expected:\n%s", tt.output)
			t.Logf("Actual:\n%s", output)
		}
	}
}
