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
	"io/ioutil"
	"testing"
)

func TestDeserializeAndReserialize(t *testing.T) {
	tests := []struct {
		in   string
		wout string
	}{
		{
			`[Service]
ExecStart=/bin/bash -c "while true; do echo \"ping\"; sleep 1; done"
`,
			`[Service]
ExecStart=/bin/bash -c "while true; do echo \"ping\"; sleep 1; done"
`},
		{
			`[Unit]
Description= Unnecessarily wrapped \
    words here`,
			`[Unit]
Description=Unnecessarily wrapped \
    words here
`,
		},
		{
			`[Unit]
Description=Demo \

Requires=docker.service
`,
			`[Unit]
Description=Demo \

Requires=docker.service
`,
		},
		{
			`; comment alpha
# comment bravo
[Unit]
; comment charlie
# comment delta
#Description=Foo
Description=Bar
; comment echo
# comment foxtrot
`,
			`[Unit]
Description=Bar
`},
	}
	for i, tt := range tests {
		ds, err := Deserialize(bytes.NewBufferString(tt.in))
		if err != nil {
			t.Errorf("case %d: unexpected error parsing unit: %v", i, err)
			continue
		}
		out, err := ioutil.ReadAll(Serialize(ds))
		if err != nil {
			t.Errorf("case %d: unexpected error serializing unit: %v", i, err)
			continue
		}
		if g := string(out); g != tt.wout {
			t.Errorf("case %d: incorrect output", i)
			t.Logf("Expected:\n%#v", tt.wout)
			t.Logf("Actual:\n%#v", g)
		}
	}
}
