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
	"reflect"
	"testing"
)

func TestAppValid(t *testing.T) {
	tests := []App{
		App{
			Exec:             []string{"/bin/httpd"},
			User:             "0",
			Group:            "0",
			WorkingDirectory: "/tmp",
		},
		App{
			Exec:  []string{"/app"},
			User:  "0",
			Group: "0",
			EventHandlers: []EventHandler{
				{Name: "pre-start"},
				{Name: "post-stop"},
			},
			Environment: []EnvironmentVariable{
				{Name: "DEBUG", Value: "true"},
			},
			WorkingDirectory: "/tmp",
		},
		App{
			Exec:             []string{"/app", "arg1", "arg2"},
			User:             "0",
			Group:            "0",
			WorkingDirectory: "/tmp",
		},
	}
	for i, tt := range tests {
		if err := tt.assertValid(); err != nil {
			t.Errorf("#%d: err == %v, want nil", i, err)
		}
	}
}

func TestAppExecInvalid(t *testing.T) {
	tests := []App{
		App{
			Exec: nil,
		},
		App{
			Exec:  []string{},
			User:  "0",
			Group: "0",
		},
		App{
			Exec:  []string{"app"},
			User:  "0",
			Group: "0",
		},
		App{
			Exec:  []string{"bin/app", "arg1"},
			User:  "0",
			Group: "0",
		},
	}
	for i, tt := range tests {
		if err := tt.assertValid(); err == nil {
			t.Errorf("#%d: err == nil, want non-nil", i)
		}
	}
}

func TestAppEventHandlersInvalid(t *testing.T) {
	tests := []App{
		App{
			Exec:  []string{"/bin/httpd"},
			User:  "0",
			Group: "0",
			EventHandlers: []EventHandler{
				EventHandler{
					Name: "pre-start",
				},
				EventHandler{
					Name: "pre-start",
				},
			},
		},
		App{
			Exec:  []string{"/bin/httpd"},
			User:  "0",
			Group: "0",
			EventHandlers: []EventHandler{
				EventHandler{
					Name: "post-stop",
				},
				EventHandler{
					Name: "pre-start",
				},
				EventHandler{
					Name: "post-stop",
				},
			},
		},
	}
	for i, tt := range tests {
		if err := tt.assertValid(); err == nil {
			t.Errorf("#%d: err == nil, want non-nil", i)
		}
	}
}

func TestUserGroupInvalid(t *testing.T) {
	tests := []App{
		App{
			Exec: []string{"/app"},
		},
		App{
			Exec: []string{"/app"},
			User: "0",
		},
		App{
			Exec:  []string{"app"},
			Group: "0",
		},
	}
	for i, tt := range tests {
		if err := tt.assertValid(); err == nil {
			t.Errorf("#%d: err == nil, want non-nil", i)
		}
	}
}

func TestAppWorkingDirectoryInvalid(t *testing.T) {
	tests := []App{
		App{
			Exec:             []string{"/app"},
			User:             "foo",
			Group:            "bar",
			WorkingDirectory: "stuff",
		},
		App{
			Exec:             []string{"/app"},
			User:             "foo",
			Group:            "bar",
			WorkingDirectory: "../home/fred",
		},
	}
	for i, tt := range tests {
		if err := tt.assertValid(); err == nil {
			t.Errorf("#%d: err == nil, want non-nil", i)
		}
	}
}

func TestAppEnvironmentInvalid(t *testing.T) {
	tests := []App{
		App{
			Exec:  []string{"/app"},
			User:  "foo",
			Group: "bar",
			Environment: Environment{
				EnvironmentVariable{"0DEBUG", "true"},
			},
		},
	}
	for i, tt := range tests {
		if err := tt.assertValid(); err == nil {
			t.Errorf("#%d: err == nil, want non-nil", i)
		}
	}
}

func TestAppUnmarshal(t *testing.T) {
	tests := []struct {
		in   string
		wann *App
		werr bool
	}{
		{
			`garbage`,
			&App{},
			true,
		},
		{
			`{"Exec":"not a list"}`,
			&App{},
			true,
		},
		{
			`{"Exec":["notfullyqualified"]}`,
			&App{},
			true,
		},
		{
			`{"Exec":["/a"],"User":"0","Group":"0"}`,
			&App{
				Exec: Exec{
					"/a",
				},
				User:        "0",
				Group:       "0",
				Environment: make(Environment, 0),
			},
			false,
		},
	}
	for i, tt := range tests {
		a := &App{}
		err := a.UnmarshalJSON([]byte(tt.in))
		gerr := err != nil
		if gerr != tt.werr {
			t.Errorf("#%d: gerr=%t, want %t (err=%v)", i, gerr, tt.werr, err)
		}
		if !reflect.DeepEqual(a, tt.wann) {
			t.Errorf("#%d: ann=%#v, want %#v", i, a, tt.wann)
		}
	}
}
