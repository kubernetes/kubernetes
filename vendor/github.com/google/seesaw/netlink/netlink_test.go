// Copyright 2015 Google Inc. All Rights Reserved.
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

package netlink

import (
	"reflect"
	"testing"
)

func TestFieldParams(t *testing.T) {
	tests := []struct {
		params  string
		want    *fieldParams
		wantErr bool
	}{
		{
			"",
			nil,
			false,
		},
		{
			"attr:123",
			&fieldParams{attr: 123},
			false,
		},
		{
			"attr:-123",
			nil,
			true,
		},
		{
			"attr:65536",
			nil,
			true,
		},
		{
			"attr:123,omitempty,optional",
			&fieldParams{attr: 123, omitempty: true, optional: true},
			false,
		},
		{
			"abcxyz",
			nil,
			true,
		},
	}
	for _, test := range tests {
		got, err := parseFieldParams(test.params)
		switch {
		case test.wantErr && err == nil:
			t.Errorf("parseFieldParams(%q): got nil error, want error", test.params)
		case !test.wantErr && err != nil:
			t.Errorf("parseFieldParams(%q) returned error: %v", test.params, err)
		case !test.wantErr && !reflect.DeepEqual(got, test.want):
			t.Errorf("parseFieldParams(%q) = %#v, want %#v", test.params, got, test.want)
		}
	}
}

func TestMaxAttrID(t *testing.T) {
	tests := []struct {
		input interface{}
		want  uint16
	}{
		{
			struct{}{},
			0,
		},
		{
			struct {
				a uint32 `netlink:"attr:65535"`
			}{},
			65535,
		},
		{
			struct {
				a uint8  `netlink:"attr:1"`
				b uint16 `netlink:"attr:2"`
				c uint32 `netlink:"attr:3"`
			}{},
			3,
		},
		{
			struct {
				c uint32 `netlink:"attr:3"`
				b uint16 `netlink:"attr:2"`
				a uint8  `netlink:"attr:1"`
			}{},
			3,
		},
		{
			struct {
				a uint8  `netlink:"attr:1"`
				b uint16 `netlink:"attr:2"`
				c uint32 `netlink:"attr:3"`
				x struct {
					d string `netlink:"attr:5"`
				}
			}{},
			5,
		},
		{
			struct {
				a uint8  `netlink:"attr:1"`
				b uint16 `netlink:"attr:2"`
				c uint32 `netlink:"attr:3"`
				x struct {
					d string `netlink:"attr:5"`
				} `netlink:"attr:4"`
			}{},
			4,
		},
	}
	for i, test := range tests {
		got, err := structMaxAttrID(reflect.ValueOf(test.input))
		if err != nil {
			t.Errorf("%d: got error: %v", i, err)
			continue
		}
		if got != test.want {
			t.Errorf("%d: got %d, want %d", i, got, test.want)
		}
	}
}
