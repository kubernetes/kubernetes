// Copyright 2015 The appc Authors
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

package types

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestLabels(t *testing.T) {
	tests := []struct {
		in        string
		errPrefix string
	}{
		{
			`[{"name": "os", "value": "linux"}, {"name": "arch", "value": "amd64"}]`,
			"",
		},
		{
			`[{"name": "os", "value": "linux"}, {"name": "arch", "value": "aarch64"}]`,
			"",
		},
		{
			`[{"name": "os", "value": "linux"}, {"name": "arch", "value": "arm64"}]`,
			`bad arch "arm64" for linux`,
		},
		{
			`[{"name": "os", "value": "linux"}, {"name": "arch", "value": "aarch64_be"}]`,
			"",
		},
		{
			`[{"name": "os", "value": "linux"}, {"name": "arch", "value": "arm64_be"}]`,
			`bad arch "arm64_be" for linux`,
		},
		{
			`[{"name": "os", "value": "linux"}, {"name": "arch", "value": "arm"}]`,
			`bad arch "arm" for linux`,
		},
		{
			`[{"name": "os", "value": "linux"}, {"name": "arch", "value": "armv6l"}]`,
			"",
		},
		{
			`[{"name": "os", "value": "linux"}, {"name": "arch", "value": "armv7l"}]`,
			"",
		},
		{
			`[{"name": "os", "value": "linux"}, {"name": "arch", "value": "armv7b"}]`,
			"",
		},
		{
			`[{"name": "os", "value": "freebsd"}, {"name": "arch", "value": "amd64"}]`,
			"",
		},
		{
			`[{"name": "os", "value": "OS/360"}, {"name": "arch", "value": "S/360"}]`,
			`bad os "OS/360"`,
		},
		{
			`[{"name": "os", "value": "freebsd"}, {"name": "arch", "value": "armv7b"}]`,
			`bad arch "armv7b" for freebsd`,
		},
		{
			`[{"name": "name"}]`,
			`invalid label name: "name"`,
		},
		{
			`[{"name": "os", "value": "linux"}, {"name": "os", "value": "freebsd"}]`,
			`duplicate labels of name "os"`,
		},
		{
			`[{"name": "arch", "value": "amd64"}, {"name": "os", "value": "freebsd"}, {"name": "arch", "value": "x86_64"}]`,
			`duplicate labels of name "arch"`,
		},
		{
			`[]`,
			"",
		},
	}
	for i, tt := range tests {
		var l Labels
		if err := json.Unmarshal([]byte(tt.in), &l); err != nil {
			if tt.errPrefix == "" {
				t.Errorf("#%d: got err=%v, expected no error", i, err)
			} else if !strings.HasPrefix(err.Error(), tt.errPrefix) {
				t.Errorf("#%d: got err=%v, expected prefix %#v", i, err, tt.errPrefix)
			}
		} else {
			t.Log(l)
			if tt.errPrefix != "" {
				t.Errorf("#%d: got no err, expected prefix %#v", i, tt.errPrefix)
			}
		}
	}
}
