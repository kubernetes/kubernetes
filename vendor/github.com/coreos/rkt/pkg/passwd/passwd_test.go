// Copyright 2016 The rkt Authors
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

package passwd

import (
	"reflect"
	"testing"
)

func TestParsePasswdLine(t *testing.T) {
	tests := []struct {
		line          string
		passwdLine    *User
		shouldSucceed bool
	}{
		{
			"nobody:x:1000:100::/home/nobody:",
			&User{
				"nobody",
				"x",
				1000,
				100,
				"",
				"/home/nobody",
				"",
			},
			true,
		},
		{
			"nobody:x:1000:100::/home/nobody:/bin/nologin",
			&User{
				"nobody",
				"x",
				1000,
				100,
				"",
				"/home/nobody",
				"/bin/nologin",
			},
			true,
		},
		{
			"nobody:x:",
			nil,
			false,
		},
		{
			"",
			nil,
			false,
		},
		{
			"nobody:x:1000:100::/home/nobody:/bin/nologin:more:stuff",
			&User{
				"nobody",
				"x",
				1000,
				100,
				"",
				"/home/nobody",
				"/bin/nologin",
			},
			true,
		},
	}

	for i, tt := range tests {
		p, err := parsePasswdLine(tt.line)
		if err != nil {
			if tt.shouldSucceed {
				t.Errorf("#%d: parsing line %q failed unexpectedly", i, tt.line)
			}
			continue
		}
		if !reflect.DeepEqual(p, tt.passwdLine) {
			t.Errorf("#%d: got user %v, want user %v", i, p, tt.passwdLine)
		}
	}
}
