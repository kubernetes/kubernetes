// Copyright 2015 The rkt Authors
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

package group

import (
	"reflect"
	"testing"
)

func TestParseGroupLine(t *testing.T) {
	tests := []struct {
		line          string
		groupLine     *Group
		shouldSucceed bool
	}{
		{
			"ftp:x:1:",
			&Group{
				"ftp",
				"x",
				1,
				[]string{},
			},
			true,
		},
		{
			"u1:xxx:12:wheel,users",
			&Group{
				"u1",
				"xxx",
				12,
				[]string{"wheel", "users"},
			},
			true,
		},
		{
			"uerr:x:",
			nil,
			false,
		},
		{
			"",
			nil,
			false,
		},
		{
			"u1:xxx:12:wheel,users:extra:stuff",
			&Group{
				"u1",
				"xxx",
				12,
				[]string{"wheel", "users"},
			},
			true,
		},
	}

	for i, tt := range tests {
		g, err := parseGroupLine(tt.line)
		if err != nil {
			if tt.shouldSucceed {
				t.Errorf("#%d: parsing line %q failed unexpectedly", i, tt.line)
			}
			continue
		}
		if !reflect.DeepEqual(g, tt.groupLine) {
			t.Errorf("#%d: got group %v, want group %v", i, g, tt.groupLine)
		}
	}
}
