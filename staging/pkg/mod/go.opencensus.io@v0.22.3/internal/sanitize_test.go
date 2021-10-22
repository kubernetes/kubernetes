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

package internal

import (
	"strings"
	"testing"
)

func TestSanitize(t *testing.T) {
	tests := []struct {
		name  string
		input string
		want  string
	}{
		{
			name:  "trunacate long string",
			input: strings.Repeat("a", 101),
			want:  strings.Repeat("a", 100),
		},
		{
			name:  "replace character",
			input: "test/key-1",
			want:  "test_key_1",
		},
		{
			name:  "add prefix if starting with digit",
			input: "0123456789",
			want:  "key_0123456789",
		},
		{
			name:  "add prefix if starting with _",
			input: "_0123456789",
			want:  "key_0123456789",
		},
		{
			name:  "starts with _ after sanitization",
			input: "/0123456789",
			want:  "key_0123456789",
		},
		{
			name:  "valid input",
			input: "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz_0123456789",
			want:  "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz_0123456789",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got, want := Sanitize(tt.input), tt.want; got != want {
				t.Errorf("sanitize() = %q; want %q", got, want)
			}
		})
	}
}
