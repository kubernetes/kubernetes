/*
Copyright 2023 The Kubernetes Authors.

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

package main_test

import (
	"bytes"
	"k8s.io/code-generator"
	"k8s.io/code-generator/internal/codegen/execution"
	"testing"
)

func TestMainFn(t *testing.T) {
	var out, err bytes.Buffer
	var retcode *int
	main.RunMain(func(ex *execution.Vars) {
		ex.Args = []string{"--help"}
		ex.Out = &out
		ex.Out = &err
		ex.Exit = func(code int) {
			retcode = &code
		}
	})

	if retcode != nil {
		t.Errorf("expected exit func will not be called, but was with %d", *retcode)
	}
	if out.Len() != 0 {
		t.Errorf("expected no output, got %#v", out.String())
	}
	if !bytes.Contains(err.Bytes(), []byte("Command:")) {
		t.Errorf("expected usage output, got %#v", err.String())
	}
}
