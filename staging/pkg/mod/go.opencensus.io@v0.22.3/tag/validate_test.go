// Copyright 2017, OpenCensus Authors
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

package tag

import (
	"strings"
	"testing"
)

func TestCheckKeyName(t *testing.T) {
	tests := []struct {
		name   string
		key    string
		wantOK bool
	}{
		{
			name:   "valid",
			key:    "k1",
			wantOK: true,
		},
		{
			name:   "invalid i",
			key:    "k\x19",
			wantOK: false,
		},
		{
			name:   "invalid ii",
			key:    "k\x7f",
			wantOK: false,
		},
		{
			name:   "empty",
			key:    "",
			wantOK: false,
		},
		{
			name:   "whitespace",
			key:    " k ",
			wantOK: true,
		},
		{
			name:   "long",
			key:    strings.Repeat("a", 256),
			wantOK: false,
		},
	}
	for _, tt := range tests {
		ok := checkKeyName(tt.key)
		if ok != tt.wantOK {
			t.Errorf("%v: got %v; want %v", tt.name, ok, tt.wantOK)
		}
	}
}

func TestCheckValue(t *testing.T) {
	tests := []struct {
		name   string
		value  string
		wantOK bool
	}{
		{
			name:   "valid",
			value:  "v1",
			wantOK: true,
		},
		{
			name:   "invalid i",
			value:  "k\x19",
			wantOK: false,
		},
		{
			name:   "invalid ii",
			value:  "k\x7f",
			wantOK: false,
		},
		{
			name:   "empty",
			value:  "",
			wantOK: true,
		},
		{
			name:   "whitespace",
			value:  " v ",
			wantOK: true,
		},
		{
			name:   "long",
			value:  strings.Repeat("a", 256),
			wantOK: false,
		},
	}
	for _, tt := range tests {
		ok := checkValue(tt.value)
		if ok != tt.wantOK {
			t.Errorf("%v: got %v; want %v", tt.name, ok, tt.wantOK)
		}
	}
}
