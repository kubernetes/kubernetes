/*
Copyright 2011 The go4 Authors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package jsonconfig

import (
	"os"
	"reflect"
	"strings"
	"testing"
)

func testIncludes(configFile string, t *testing.T) {
	var c ConfigParser
	c.IncludeDirs = []string{"testdata"}
	obj, err := c.ReadFile(configFile)
	if err != nil {
		t.Fatal(err)
	}
	two := obj.RequiredObject("two")
	if err := obj.Validate(); err != nil {
		t.Error(err)
	}
	if g, e := two.RequiredString("key"), "value"; g != e {
		t.Errorf("sub object key = %q; want %q", g, e)
	}
}

func TestIncludesCWD(t *testing.T) {
	testIncludes("testdata/include1.json", t)
}

func TestIncludesIncludeDirs(t *testing.T) {
	testIncludes("testdata/include1bis.json", t)
}

func TestIncludeLoop(t *testing.T) {
	_, err := ReadFile("testdata/loop1.json")
	if err == nil {
		t.Fatal("expected an error about import cycles.")
	}
	if !strings.Contains(err.Error(), "include cycle detected") {
		t.Fatalf("expected an error about import cycles; got: %v", err)
	}
}

func TestBoolEnvs(t *testing.T) {
	os.Setenv("TEST_EMPTY", "")
	os.Setenv("TEST_TRUE", "true")
	os.Setenv("TEST_ONE", "1")
	os.Setenv("TEST_ZERO", "0")
	os.Setenv("TEST_FALSE", "false")
	obj, err := ReadFile("testdata/boolenv.json")
	if err != nil {
		t.Fatal(err)
	}
	if str := obj.RequiredString("emptystr"); str != "" {
		t.Errorf("str = %q, want empty", str)
	}
	tests := []struct {
		key  string
		want bool
	}{
		{"def_false", false},
		{"def_true", true},
		{"set_true_def_false", true},
		{"set_false_def_true", false},
		{"lit_true", true},
		{"lit_false", false},
		{"one", true},
		{"zero", false},
	}
	for _, tt := range tests {
		if v := obj.RequiredBool(tt.key); v != tt.want {
			t.Errorf("key %q = %v; want %v", tt.key, v, tt.want)
		}
	}
	if err := obj.Validate(); err != nil {
		t.Error(err)
	}
}

func TestListExpansion(t *testing.T) {
	os.Setenv("TEST_BAR", "bar")
	obj, err := ReadFile("testdata/listexpand.json")
	if err != nil {
		t.Fatal(err)
	}
	s := obj.RequiredString("str")
	l := obj.RequiredList("list")
	if err := obj.Validate(); err != nil {
		t.Error(err)
	}
	want := []string{"foo", "bar"}
	if !reflect.DeepEqual(l, want) {
		t.Errorf("got = %#v\nwant = %#v", l, want)
	}
	if s != "bar" {
		t.Errorf("str = %q, want %q", s, "bar")
	}
}
