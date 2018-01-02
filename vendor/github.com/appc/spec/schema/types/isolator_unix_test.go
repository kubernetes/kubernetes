// Copyright 2016 The appc Authors
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
	"reflect"
	"testing"
)

func TestUnixSysctlIsolator(t *testing.T) {
	tests := []struct {
		inCfg map[string]string

		expectedErr bool
		expectedRes UnixSysctl
	}{
		// empty isolator - valid
		{
			make(map[string]string),

			false,
			UnixSysctl{},
		},
		// simple isolator - valid
		{
			map[string]string{
				"foo": "bar",
			},

			false,
			UnixSysctl{
				"foo": "bar",
			},
		},
	}
	for i, tt := range tests {
		gotRes, err := NewUnixSysctlIsolator(tt.inCfg)
		if gotErr := err != nil; gotErr != tt.expectedErr {
			t.Errorf("#%d: want err=%t, got %t (err=%v)", i, tt.expectedErr, gotErr, err)
		}
		if !reflect.DeepEqual(tt.expectedRes, *gotRes) {
			t.Errorf("#%d: want %s, got %s", i, tt.expectedRes, *gotRes)
		}
	}
}
