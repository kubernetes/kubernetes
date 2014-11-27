// Copyright 2014 go-dockerclient authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the DOCKER-LICENSE file.

package docker

import (
	"bytes"
	"errors"
	"reflect"
	"sort"
	"testing"
)

func TestGet(t *testing.T) {
	var tests = []struct {
		input    []string
		query    string
		expected string
	}{
		{[]string{"PATH=/usr/bin:/bin", "PYTHONPATH=/usr/local"}, "PATH", "/usr/bin:/bin"},
		{[]string{"PATH=/usr/bin:/bin", "PYTHONPATH=/usr/local"}, "PYTHONPATH", "/usr/local"},
		{[]string{"PATH=/usr/bin:/bin", "PYTHONPATH=/usr/local"}, "PYTHONPATHI", ""},
		{[]string{"WAT="}, "WAT", ""},
	}
	for _, tt := range tests {
		env := Env(tt.input)
		got := env.Get(tt.query)
		if got != tt.expected {
			t.Errorf("Env.Get(%q): wrong result. Want %q. Got %q", tt.query, tt.expected, got)
		}
	}
}

func TestExists(t *testing.T) {
	var tests = []struct {
		input    []string
		query    string
		expected bool
	}{
		{[]string{"WAT=", "PYTHONPATH=/usr/local"}, "WAT", true},
		{[]string{"PATH=/usr/bin:/bin", "PYTHONPATH=/usr/local"}, "PYTHONPATH", true},
		{[]string{"PATH=/usr/bin:/bin", "PYTHONPATH=/usr/local"}, "PYTHONPATHI", false},
	}
	for _, tt := range tests {
		env := Env(tt.input)
		got := env.Exists(tt.query)
		if got != tt.expected {
			t.Errorf("Env.Exists(%q): wrong result. Want %v. Got %v", tt.query, tt.expected, got)
		}
	}
}

func TestGetBool(t *testing.T) {
	var tests = []struct {
		input    string
		expected bool
	}{
		{"EMTPY_VAR", false}, {"ZERO_VAR", false}, {"NO_VAR", false},
		{"FALSE_VAR", false}, {"NONE_VAR", false}, {"TRUE_VAR", true},
		{"WAT", true}, {"PATH", true}, {"ONE_VAR", true}, {"NO_VAR_TAB", false},
	}
	env := Env([]string{
		"EMPTY_VAR=", "ZERO_VAR=0", "NO_VAR=no", "FALSE_VAR=false",
		"NONE_VAR=none", "TRUE_VAR=true", "WAT=wat", "PATH=/usr/bin:/bin",
		"ONE_VAR=1", "NO_VAR_TAB=0 \t\t\t",
	})
	for _, tt := range tests {
		got := env.GetBool(tt.input)
		if got != tt.expected {
			t.Errorf("Env.GetBool(%q): wrong result. Want %v. Got %v.", tt.input, tt.expected, got)
		}
	}
}

func TestSetBool(t *testing.T) {
	var tests = []struct {
		input    bool
		expected string
	}{
		{true, "1"}, {false, "0"},
	}
	for _, tt := range tests {
		var env Env
		env.SetBool("SOME", tt.input)
		if got := env.Get("SOME"); got != tt.expected {
			t.Errorf("Env.SetBool(%v): wrong result. Want %q. Got %q", tt.input, tt.expected, got)
		}
	}
}

func TestGetInt(t *testing.T) {
	var tests = []struct {
		input    string
		expected int
	}{
		{"NEGATIVE_INTEGER", -10}, {"NON_INTEGER", -1}, {"ONE", 1}, {"TWO", 2},
	}
	env := Env([]string{"NEGATIVE_INTEGER=-10", "NON_INTEGER=wat", "ONE=1", "TWO=2"})
	for _, tt := range tests {
		got := env.GetInt(tt.input)
		if got != tt.expected {
			t.Errorf("Env.GetInt(%q): wrong result. Want %d. Got %d", tt.input, tt.expected, got)
		}
	}
}

func TestSetInt(t *testing.T) {
	var tests = []struct {
		input    int
		expected string
	}{
		{10, "10"}, {13, "13"}, {7, "7"}, {33, "33"},
		{0, "0"}, {-34, "-34"},
	}
	for _, tt := range tests {
		var env Env
		env.SetInt("SOME", tt.input)
		if got := env.Get("SOME"); got != tt.expected {
			t.Errorf("Env.SetBool(%d): wrong result. Want %q. Got %q", tt.input, tt.expected, got)
		}
	}
}

func TestGetInt64(t *testing.T) {
	var tests = []struct {
		input    string
		expected int64
	}{
		{"NEGATIVE_INTEGER", -10}, {"NON_INTEGER", -1}, {"ONE", 1}, {"TWO", 2},
	}
	env := Env([]string{"NEGATIVE_INTEGER=-10", "NON_INTEGER=wat", "ONE=1", "TWO=2"})
	for _, tt := range tests {
		got := env.GetInt64(tt.input)
		if got != tt.expected {
			t.Errorf("Env.GetInt64(%q): wrong result. Want %d. Got %d", tt.input, tt.expected, got)
		}
	}
}

func TestSetInt64(t *testing.T) {
	var tests = []struct {
		input    int64
		expected string
	}{
		{10, "10"}, {13, "13"}, {7, "7"}, {33, "33"},
		{0, "0"}, {-34, "-34"},
	}
	for _, tt := range tests {
		var env Env
		env.SetInt64("SOME", tt.input)
		if got := env.Get("SOME"); got != tt.expected {
			t.Errorf("Env.SetBool(%d): wrong result. Want %q. Got %q", tt.input, tt.expected, got)
		}
	}
}

func TestGetJSON(t *testing.T) {
	var p struct {
		Name string `json:"name"`
		Age  int    `json:"age"`
	}
	var env Env
	env.Set("person", `{"name":"Gopher","age":5}`)
	err := env.GetJSON("person", &p)
	if err != nil {
		t.Error(err)
	}
	if p.Name != "Gopher" {
		t.Errorf("Env.GetJSON(%q): wrong name. Want %q. Got %q", "person", "Gopher", p.Name)
	}
	if p.Age != 5 {
		t.Errorf("Env.GetJSON(%q): wrong age. Want %d. Got %d", "person", 5, p.Age)
	}
}

func TestGetJSONAbsent(t *testing.T) {
	var l []string
	var env Env
	err := env.GetJSON("person", &l)
	if err != nil {
		t.Error(err)
	}
	if l != nil {
		t.Errorf("Env.GetJSON(): get unexpected list %v", l)
	}
}

func TestGetJSONFailure(t *testing.T) {
	var p []string
	var env Env
	env.Set("list-person", `{"name":"Gopher","age":5}`)
	err := env.GetJSON("list-person", &p)
	if err == nil {
		t.Errorf("Env.GetJSON(%q): got unexpected <nil> error.", "list-person")
	}
}

func TestSetJSON(t *testing.T) {
	var p1 = struct {
		Name string `json:"name"`
		Age  int    `json:"age"`
	}{Name: "Gopher", Age: 5}
	var env Env
	err := env.SetJSON("person", p1)
	if err != nil {
		t.Error(err)
	}
	var p2 struct {
		Name string `json:"name"`
		Age  int    `json:"age"`
	}
	err = env.GetJSON("person", &p2)
	if err != nil {
		t.Error(err)
	}
	if !reflect.DeepEqual(p1, p2) {
		t.Errorf("Env.SetJSON(%q): wrong result. Want %v. Got %v", "person", p1, p2)
	}
}

func TestSetJSONFailure(t *testing.T) {
	var env Env
	err := env.SetJSON("person", unmarshable{})
	if err == nil {
		t.Error("Env.SetJSON(): got unexpected <nil> error")
	}
	if env.Exists("person") {
		t.Errorf("Env.SetJSON(): should not define the key %q, but did", "person")
	}
}

func TestGetList(t *testing.T) {
	var tests = []struct {
		input    string
		expected []string
	}{
		{"WAT=wat", []string{"wat"}},
		{`WAT=["wat","wet","wit","wot","wut"]`, []string{"wat", "wet", "wit", "wot", "wut"}},
		{"WAT=", nil},
	}
	for _, tt := range tests {
		env := Env([]string{tt.input})
		got := env.GetList("WAT")
		if !reflect.DeepEqual(got, tt.expected) {
			t.Errorf("Env.GetList(%q): wrong result. Want %v. Got %v", "WAT", tt.expected, got)
		}
	}
}

func TestSetList(t *testing.T) {
	list := []string{"a", "b", "c"}
	var env Env
	env.SetList("SOME", list)
	if got := env.GetList("SOME"); !reflect.DeepEqual(got, list) {
		t.Errorf("Env.SetList(%v): wrong result. Got %v", list, got)
	}
}

func TestSet(t *testing.T) {
	var env Env
	env.Set("PATH", "/home/bin:/bin")
	env.Set("SOMETHING", "/usr/bin")
	env.Set("PATH", "/bin")
	if expected, got := "/usr/bin", env.Get("SOMETHING"); got != expected {
		t.Errorf("Env.Set(%q): wrong result. Want %q. Got %q", expected, expected, got)
	}
	if expected, got := "/bin", env.Get("PATH"); got != expected {
		t.Errorf("Env.Set(%q): wrong result. Want %q. Got %q", expected, expected, got)
	}
}

func TestDecode(t *testing.T) {
	var tests = []struct {
		input       string
		expectedOut []string
		expectedErr string
	}{
		{
			`{"PATH":"/usr/bin:/bin","containers":54,"wat":["123","345"]}`,
			[]string{"PATH=/usr/bin:/bin", "containers=54", `wat=["123","345"]`},
			"",
		},
		{"}}", nil, "invalid character '}' looking for beginning of value"},
		{`{}`, nil, ""},
	}
	for _, tt := range tests {
		var env Env
		err := env.Decode(bytes.NewBufferString(tt.input))
		if tt.expectedErr == "" {
			if err != nil {
				t.Error(err)
			}
		} else if tt.expectedErr != err.Error() {
			t.Errorf("Env.Decode(): invalid error. Want %q. Got %q.", tt.expectedErr, err)
		}
		got := []string(env)
		sort.Strings(got)
		sort.Strings(tt.expectedOut)
		if !reflect.DeepEqual(got, tt.expectedOut) {
			t.Errorf("Env.Decode(): wrong result. Want %v. Got %v.", tt.expectedOut, got)
		}
	}
}

func TestSetAuto(t *testing.T) {
	buf := bytes.NewBufferString("oi")
	var tests = []struct {
		input    interface{}
		expected string
	}{
		{10, "10"},
		{10.3, "10"},
		{"oi", "oi"},
		{buf, "{}"},
		{unmarshable{}, "{}"},
	}
	for _, tt := range tests {
		var env Env
		env.SetAuto("SOME", tt.input)
		if got := env.Get("SOME"); got != tt.expected {
			t.Errorf("Env.SetAuto(%v): wrong result. Want %q. Got %q", tt.input, tt.expected, got)
		}
	}
}

func TestMap(t *testing.T) {
	var tests = []struct {
		input    []string
		expected map[string]string
	}{
		{[]string{"PATH=/usr/bin:/bin", "PYTHONPATH=/usr/local"}, map[string]string{"PATH": "/usr/bin:/bin", "PYTHONPATH": "/usr/local"}},
		{nil, nil},
	}
	for _, tt := range tests {
		env := Env(tt.input)
		got := env.Map()
		if !reflect.DeepEqual(got, tt.expected) {
			t.Errorf("Env.Map(): wrong result. Want %v. Got %v", tt.expected, got)
		}
	}
}

type unmarshable struct {
}

func (unmarshable) MarshalJSON() ([]byte, error) {
	return nil, errors.New("cannot marshal")
}
