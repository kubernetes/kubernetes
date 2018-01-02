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

package discovery

import (
	"reflect"
	"testing"

	"github.com/appc/spec/schema/types"
)

func TestNewAppFromString(t *testing.T) {
	tests := []struct {
		in string

		w    *App
		werr bool
	}{
		{
			"example.com/reduce-worker:1.0.0",

			&App{
				Name: "example.com/reduce-worker",
				Labels: map[types.ACIdentifier]string{
					"version": "1.0.0",
				},
			},
			false,
		},
		{
			"example.com/reduce-worker,channel=alpha,label=value",

			&App{
				Name: "example.com/reduce-worker",
				Labels: map[types.ACIdentifier]string{
					"channel": "alpha",
					"label":   "value",
				},
			},

			false,
		},
		{
			"example.com/app:1.2.3,special=!*'();@&+$/?#[],channel=beta",

			&App{
				Name: "example.com/app",
				Labels: map[types.ACIdentifier]string{
					"version": "1.2.3",
					"special": "!*'();@&+$/?#[]",
					"channel": "beta",
				},
			},

			false,
		},

		// bad AC name for app
		{
			"not an app name",

			nil,
			true,
		},

		// bad URL escape (bad query)
		{
			"example.com/garbage%8 939",

			nil,
			true,
		},

		// multi-value labels
		{
			"foo.com/bar,channel=alpha,dog=woof,channel=beta",

			nil,
			true,
		},
		// colon coming after some label instead of being
		// right after the name
		{
			"example.com/app,channel=beta:1.2.3",

			nil,
			true,
		},
		// two colons in string
		{
			"example.com/app:3.2.1,channel=beta:1.2.3",

			nil,
			true,
		},
		// two version labels, one implicit, one explicit
		{
			"example.com/app:3.2.1,version=1.2.3",

			nil,
			true,
		},
	}
	for i, tt := range tests {
		g, err := NewAppFromString(tt.in)
		gerr := (err != nil)
		if !reflect.DeepEqual(g, tt.w) {
			t.Errorf("#%d: got %v, want %v", i, g, tt.w)
		}
		if gerr != tt.werr {
			t.Errorf("#%d: gerr=%t, want %t (err=%v)", i, gerr, tt.werr, err)
		}
	}
}

func TestAppString(t *testing.T) {
	tests := []struct {
		a   *App
		out string
	}{
		{
			&App{
				Name:   "example.com/reduce-worker",
				Labels: map[types.ACIdentifier]string{},
			},
			"example.com/reduce-worker",
		},
		{
			&App{
				Name: "example.com/reduce-worker",
				Labels: map[types.ACIdentifier]string{
					"version": "1.0.0",
				},
			},
			"example.com/reduce-worker:1.0.0",
		},
		{
			&App{
				Name: "example.com/reduce-worker",
				Labels: map[types.ACIdentifier]string{
					"channel": "alpha",
					"label":   "value",
				},
			},
			"example.com/reduce-worker,channel=alpha,label=value",
		},
	}
	for i, tt := range tests {
		appString := tt.a.String()

		g, err := NewAppFromString(appString)
		if err != nil {
			t.Errorf("unexpected error: %v", err)
		}
		if !reflect.DeepEqual(g, tt.a) {
			t.Errorf("#%d: got %#v, want %#v", i, g, tt.a)
		}
	}
}

func TestAppCopy(t *testing.T) {
	tests := []struct {
		a   *App
		out string
	}{
		{
			&App{
				Name:   "example.com/reduce-worker",
				Labels: map[types.ACIdentifier]string{},
			},
			"example.com/reduce-worker",
		},
		{
			&App{
				Name: "example.com/reduce-worker",
				Labels: map[types.ACIdentifier]string{
					"version": "1.0.0",
				},
			},
			"example.com/reduce-worker:1.0.0",
		},
		{
			&App{
				Name: "example.com/reduce-worker",
				Labels: map[types.ACIdentifier]string{
					"channel": "alpha",
					"label":   "value",
				},
			},
			"example.com/reduce-worker,channel=alpha,label=value",
		},
	}
	for i, tt := range tests {
		appCopy := tt.a.Copy()
		if !reflect.DeepEqual(appCopy, tt.a) {
			t.Errorf("#%d: got %#v, want %#v", i, appCopy, tt.a)
		}
	}
}
