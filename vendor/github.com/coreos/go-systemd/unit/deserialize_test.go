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
	"bytes"
	"fmt"
	"reflect"
	"testing"
)

func TestDeserialize(t *testing.T) {
	tests := []struct {
		input  []byte
		output []*UnitOption
	}{
		// multiple options underneath a section
		{
			[]byte(`[Unit]
Description=Foo
Description=Bar
Requires=baz.service
After=baz.service
`),
			[]*UnitOption{
				&UnitOption{"Unit", "Description", "Foo"},
				&UnitOption{"Unit", "Description", "Bar"},
				&UnitOption{"Unit", "Requires", "baz.service"},
				&UnitOption{"Unit", "After", "baz.service"},
			},
		},

		// multiple sections
		{
			[]byte(`[Unit]
Description=Foo

[Service]
ExecStart=/usr/bin/sleep infinity

[X-Third-Party]
Pants=on

`),
			[]*UnitOption{
				&UnitOption{"Unit", "Description", "Foo"},
				&UnitOption{"Service", "ExecStart", "/usr/bin/sleep infinity"},
				&UnitOption{"X-Third-Party", "Pants", "on"},
			},
		},

		// multiple sections with no options
		{
			[]byte(`[Unit]
[Service]
[X-Third-Party]
`),
			[]*UnitOption{},
		},

		// multiple values not special-cased
		{
			[]byte(`[Service]
Environment= "FOO=BAR" "BAZ=QUX"
`),
			[]*UnitOption{
				&UnitOption{"Service", "Environment", "\"FOO=BAR\" \"BAZ=QUX\""},
			},
		},

		// line continuations unmodified
		{
			[]byte(`[Unit]
Description= Unnecessarily wrapped \
    words here
`),
			[]*UnitOption{
				&UnitOption{"Unit", "Description", `Unnecessarily wrapped \
    words here`},
			},
		},

		// comments ignored
		{
			[]byte(`; comment alpha
# comment bravo
[Unit]
; comment charlie
# comment delta
#Description=Foo
Description=Bar
; comment echo
# comment foxtrot
`),
			[]*UnitOption{
				&UnitOption{"Unit", "Description", "Bar"},
			},
		},

		// apparent comment lines inside of line continuations not ignored
		{
			[]byte(`[Unit]
Description=Bar\
# comment alpha

Description=Bar\
# comment bravo \
Baz
`),
			[]*UnitOption{
				&UnitOption{"Unit", "Description", "Bar\\\n# comment alpha"},
				&UnitOption{"Unit", "Description", "Bar\\\n# comment bravo \\\nBaz"},
			},
		},

		// options outside of sections are ignored
		{
			[]byte(`Description=Foo
[Unit]
Description=Bar
`),
			[]*UnitOption{
				&UnitOption{"Unit", "Description", "Bar"},
			},
		},

		// garbage outside of sections are ignored
		{
			[]byte(`<<<<<<<<
[Unit]
Description=Bar
`),
			[]*UnitOption{
				&UnitOption{"Unit", "Description", "Bar"},
			},
		},

		// garbage used as unit option
		{
			[]byte(`[Unit]
<<<<<<<<=Bar
`),
			[]*UnitOption{
				&UnitOption{"Unit", "<<<<<<<<", "Bar"},
			},
		},

		// option name with spaces are valid
		{
			[]byte(`[Unit]
Some Thing = Bar
`),
			[]*UnitOption{
				&UnitOption{"Unit", "Some Thing", "Bar"},
			},
		},

		// lack of trailing newline doesn't cause problem for non-continued file
		{
			[]byte(`[Unit]
Description=Bar`),
			[]*UnitOption{
				&UnitOption{"Unit", "Description", "Bar"},
			},
		},

		// unit file with continuation but no following line is ok, too
		{
			[]byte(`[Unit]
Description=Bar \`),
			[]*UnitOption{
				&UnitOption{"Unit", "Description", "Bar \\"},
			},
		},

		// Assert utf8 characters are preserved
		{
			[]byte(`[©]
µ☃=ÇôrèÕ$`),
			[]*UnitOption{
				&UnitOption{"©", "µ☃", "ÇôrèÕ$"},
			},
		},

		// whitespace removed around option name
		{
			[]byte(`[Unit]
 Description   =words here
`),
			[]*UnitOption{
				&UnitOption{"Unit", "Description", "words here"},
			},
		},

		// whitespace around option value stripped
		{
			[]byte(`[Unit]
Description= words here `),
			[]*UnitOption{
				&UnitOption{"Unit", "Description", "words here"},
			},
		},

		// whitespace around option value stripped, regardless of continuation
		{
			[]byte(`[Unit]
Description= words here \
  `),
			[]*UnitOption{
				&UnitOption{"Unit", "Description", "words here \\\n"},
			},
		},

		// backslash not considered continuation if followed by text
		{
			[]byte(`[Service]
ExecStart=/bin/bash -c "while true; do echo \"ping\"; sleep 1; done"
`),
			[]*UnitOption{
				&UnitOption{"Service", "ExecStart", `/bin/bash -c "while true; do echo \"ping\"; sleep 1; done"`},
			},
		},

		// backslash not considered continuation if followed by whitespace, but still trimmed
		{
			[]byte(`[Service]
ExecStart=/bin/bash echo poof \  `),
			[]*UnitOption{
				&UnitOption{"Service", "ExecStart", `/bin/bash echo poof \`},
			},
		},
		// a long unit file line that's just equal to the maximum permitted length
		{
			[]byte(`[Service]
ExecStart=/bin/bash -c "echo ................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................."`),
			[]*UnitOption{
				&UnitOption{"Service", "ExecStart", `/bin/bash -c "echo ................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................."`},
			},
		},
		// the same, but with a trailing newline
		{
			[]byte(`[Service]
ExecStart=/bin/bash -c "echo ................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................."
Option=value
`),
			[]*UnitOption{
				&UnitOption{"Service", "ExecStart", `/bin/bash -c "echo ................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................."`},
				&UnitOption{"Service", "Option", "value"},
			},
		},
	}

	assert := func(expect, output []*UnitOption) error {
		if len(expect) != len(output) {
			return fmt.Errorf("expected %d items, got %d", len(expect), len(output))
		}

		for i, _ := range expect {
			if !reflect.DeepEqual(expect[i], output[i]) {
				return fmt.Errorf("item %d: expected %v, got %v", i, expect[i], output[i])
			}
		}

		return nil
	}

	for i, tt := range tests {
		output, err := Deserialize(bytes.NewReader(tt.input))
		if err != nil {
			t.Errorf("case %d: unexpected error parsing unit: %v", i, err)
			continue
		}

		err = assert(tt.output, output)
		if err != nil {
			t.Errorf("case %d: %v", i, err)
			t.Log("Expected options:")
			logUnitOptionSlice(t, tt.output)
			t.Log("Actual options:")
			logUnitOptionSlice(t, output)
		}
	}
}

func TestDeserializeFail(t *testing.T) {
	tests := [][]byte{
		// malformed section header
		[]byte(`[Unit
Description=Foo
`),

		// garbage following section header
		[]byte(`[Unit] pants
Description=Foo
`),

		// option without value
		[]byte(`[Unit]
Description
`),

		// garbage inside of section
		[]byte(`[Unit]
<<<<<<
Description=Foo
`),
	}

	for i, tt := range tests {
		output, err := Deserialize(bytes.NewReader(tt))
		if err == nil {
			t.Errorf("case %d: unexpected nil error", i)
			t.Log("Output:")
			logUnitOptionSlice(t, output)
		}
	}
}

func logUnitOptionSlice(t *testing.T, opts []*UnitOption) {
	for idx, opt := range opts {
		t.Logf("%d: %v", idx, opt)
	}
}

func TestDeserializeLineTooLong(t *testing.T) {
	tests := [][]byte{
		// section header that's far too long
		[]byte(`[Seeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeervice]
`),
		// sane-looking unit file with a line just greater than the maximum allowed (currently, 2048)
		[]byte(`[Service]
ExecStart=/bin/bash -c "echo ..................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................."
`),
		// sane-looking unit file with option value way too long
		[]byte(`
# test unit file

[Service]
ExecStartPre=-/usr/bin/docker rm %p
ExecStartPre=-/usr/bin/docker pull busybox
ExecStart=/usr/bin/docker run --rm --name %p --net=host \
  -e "test=1123t" \
  -e "test=1123t" \
  -e "fiz=1123t" \
  -e "buz=1123t" \
  -e "FOO=BARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBABARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARRBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBAR"BARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBABARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARRBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBARBAR" \
  busybox sleep 10
ExecStop=-/usr/bin/docker kill %p
SyslogIdentifier=busybox
Restart=always
RestartSec=10s
`),
		// single arbitrary line that's way too long
		[]byte(`arbitrary and extraordinarily long line that is far greater than 2048 characters arbitrary and extraordinarily long line that is far greater than 2048 character arbitrary and extraordinarily long line that is far greater than 2048 characters arbitrary and extraordinarily long line that is far greater than 2048 characters arbitrary and extraordinarily long line that is far greater than 2048 characters arbitrary and extraordinarily long line that is far greater than 2048 characters arbitrary and extraordinarily long line that is far greater than 2048 characters arbitrary and extraordinarily long line that is far greater than 2048 characters arbitrary and extraordinarily long line that is far greater than 2048 characters arbitrary and extraordinarily long line that is far greater than 2048 characters arbitrary and extraordinarily long line that is far greater than 2048 characters arbitrary and extraordinarily long line that is far greater than 2048 characters arbitrary and extraordinarily long line that is far greater than 2048 characters arbitrary and extraordinarily long line that is far greater than 2048 characters arbitrary and extraordinarily long line that is far greater than 2048 characters arbitrary and extraordinarily long line that is far greater than 2048 character arbitrary and extraordinarily long line that is far greater than 2048 characters arbitrary and extraordinarily long line that is far greater than 2048 characters arbitrary and extraordinarily long line that is far greater than 2048 characters arbitrary and extraordinarily long line that is far greater than 2048 characters arbitrary and extraordinarily long line that is far greater than 2048 characters arbitrary and extraordinarily long line that is far greater than 2048 characters arbitrary and extraordinarily long line that is far greater than 2048 characters arbitrary and extraordinarily long line that is far greater than 2048 characters arbitrary and extraordinarily long line that is far greater than 2048 characters arbitrary and extraordinarily long line that is far greater than 2048 characters arbitrary and extraordinarily long line that is far greater than 2048 characters arbitrary and extraordinarily long line that is far greater than 2048 characters arbitrary and extraordinarily long line that is far greater than 2048 characters arbitrary and extraordinarily long line that is far greater than 2048 character arbitrary and extraordinarily long line that is far greater than 2048 characters arbitrary and extraordinarily long line that is far greater than 2048 characters arbitrary and extraordinarily long line that is far greater than 2048 characters arbitrary and extraordinarily long line that is far greater than 2048 characters arbitrary and extraordinarily long line that is far greater than 2048 characters arbitrary and extraordinarily long line that is far greater than 2048 characters arbitrary and extraordinarily long line that is far greater than 2048 characters arbitrary and extraordinarily long line that is far greater than 2048 characters arbitrary and extraordinarily long line that is far greater than 2048 characters arbitrary and extraordinarily long line that is far greater than 2048 characters arbitrary and extraordinarily long line that is far greater than 2048 characters arbitrary and extraordinarily long line that is far greater than 2048 characters arbitrary and extraordinarily long line that is far greater than 2048 characters arbitrary and extraordinarily long line that is far greater than 2048 character arbitrary and extraordinarily long line that is far greater than 2048 characters arbitrary and extraordinarily long line that is far greater than 2048 characters arbitrary and extraordinarily long line that is far greater than 2048 characters arbitrary and extraordinarily long line that is far greater than 2048 characters arbitrary and extraordinarily long line that is far greater than 2048 characters arbitrary and extraordinarily long line that is far greater than 2048 characters arbitrary and extraordinarily long line that is far greater than 2048 characters arbitrary and extraordinarily long line that is far greater than 2048 characters arbitrary and extraordinarily long line that is far greater than 2048 characters arbitrary and extraordinarily long line that is far greater than 2048 characters arbitrary and extraordinarily long line that is far greater than 2048 characters arbitrary and extraordinarily long line that is far greater than 2048 characters`),
		// sane-looking unit file with option name way too long
		[]byte(`[Service]
ExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStartExecStart=/bin/true
`),
	}

	for i, tt := range tests {
		output, err := Deserialize(bytes.NewReader(tt))
		if err != ErrLineTooLong {
			t.Errorf("case %d: unexpected err: %v", i, err)
			t.Log("Output:")
			logUnitOptionSlice(t, output)
		}
	}
}
