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

import "testing"

func TestExecValid(t *testing.T) {
	tests := []Exec{
		Exec{"/bin/httpd"},
		Exec{"/app"},
		Exec{"/app", "arg1", "arg2"},
	}
	for i, tt := range tests {
		if err := tt.assertValid(); err != nil {
			t.Errorf("#%d: err == %v, want nil", i, err)
		}
	}
}

func TestExecInvalid(t *testing.T) {
	tests := []Exec{
		Exec{},
		Exec{"app"},
	}
	for i, tt := range tests {
		if err := tt.assertValid(); err == nil {
			t.Errorf("#%d: err == nil, want non-nil", i)
		}
	}
}
