// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package internal contains support packages for oauth2 package.
package internal

import (
	"reflect"
	"strings"
	"testing"
)

func TestParseINI(t *testing.T) {
	tests := []struct {
		ini  string
		want map[string]map[string]string
	}{
		{
			`root = toor
[foo]  
bar = hop
ini = nin
`,
			map[string]map[string]string{
				"":    map[string]string{"root": "toor"},
				"foo": map[string]string{"bar": "hop", "ini": "nin"},
			},
		},
		{
			`[empty]
[section]
empty=
`,
			map[string]map[string]string{
				"":        map[string]string{},
				"empty":   map[string]string{},
				"section": map[string]string{"empty": ""},
			},
		},
		{
			`ignore
[invalid
=stuff
;comment=true
`,
			map[string]map[string]string{
				"": map[string]string{},
			},
		},
	}
	for _, tt := range tests {
		result, err := ParseINI(strings.NewReader(tt.ini))
		if err != nil {
			t.Errorf("ParseINI(%q) error %v, want: no error", tt.ini, err)
			continue
		}
		if !reflect.DeepEqual(result, tt.want) {
			t.Errorf("ParseINI(%q) = %#v, want: %#v", tt.ini, result, tt.want)
		}
	}
}
