/*
Copyright 2016 The Kubernetes Authors.

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

package main

import (
	"flag"
	"os"
	"path/filepath"
	"testing"

	"golang.org/x/tools/go/packages"
)

// This exists because `go` is not always in the PATH when running CI.
var goBinary = flag.String("go", "", "path to a `go` binary")

func TestVerify(t *testing.T) {
	// x/tools/packages is going to literally exec `go`, so it needs some
	// setup.
	setEnvVars(t)

	tcs := []struct {
		path   string
		expect int
	}{
		{"./testdata/good", 0},
		{"./testdata/bad", 18},
	}

	for _, tc := range tcs {
		errs, err := verify("linux/amd64", []string{tc.path}, nil)
		if err != nil {
			t.Errorf("unexpected error: %v", err)
		} else if len(errs) != tc.expect {
			t.Errorf("Expected %d errors, got %d: %v", tc.expect, len(errs), errs)
		}
	}
}

func setEnvVars(t testing.TB) {
	t.Helper()
	if *goBinary != "" {
		newPath := filepath.Dir(*goBinary)
		curPath := os.Getenv("PATH")
		if curPath != "" {
			newPath = newPath + ":" + curPath
		}
		t.Setenv("PATH", newPath)
	}
	if os.Getenv("HOME") == "" {
		t.Setenv("HOME", "/tmp")
	}
}

func TestDedup(t *testing.T) {
	testcases := []struct {
		input    []packages.Error
		expected int
	}{{
		input:    nil,
		expected: 0,
	}, {
		input: []packages.Error{
			{Pos: "file:7", Msg: "message", Kind: packages.ParseError},
		},
		expected: 1,
	}, {
		input: []packages.Error{
			{Pos: "file:7", Msg: "message1", Kind: packages.ParseError},
			{Pos: "file:8", Msg: "message2", Kind: packages.ParseError},
		},
		expected: 2,
	}, {
		input: []packages.Error{
			{Pos: "file:7", Msg: "message1", Kind: packages.ParseError},
			{Pos: "file:8", Msg: "message2", Kind: packages.ParseError},
			{Pos: "file:7", Msg: "message1", Kind: packages.ParseError},
		},
		expected: 2,
	}}

	for i, tc := range testcases {
		out := dedup(tc.input)
		if len(out) != tc.expected {
			t.Errorf("[%d] dedup(%v) = '%v', expected %d",
				i, tc.input, out, tc.expected)
		}
	}
}
