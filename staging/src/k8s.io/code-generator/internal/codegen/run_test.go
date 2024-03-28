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

package codegen_test

import (
	"bytes"
	"k8s.io/code-generator/internal/codegen"
	"k8s.io/code-generator/internal/codegen/execution"
	"math"
	"testing"
)

func TestRun(t *testing.T) {
	t.Parallel()
	var err bytes.Buffer
	codegen.Run(func(v *execution.Vars) {
		v.Args = []string{"help", "gen-client"}
		v.Out = &err
	})
	if !bytes.Contains(err.Bytes(), []byte("Usage: code-generator gen-client [options]")) {
		t.Errorf("Expected usage, got: %#v", err.String())
	}
}

func TestInvalid(t *testing.T) {
	t.Parallel()
	var err bytes.Buffer
	retcode := math.MinInt32
	codegen.Run(func(ex *execution.Vars) {
		ex.Args = []string{"invalid"}
		ex.Out = &err
		ex.Exit = func(code int) {
			retcode = code
		}
	})
	if !bytes.Contains(err.Bytes(), []byte("Invalid arguments given: invalid")) {
		t.Errorf("Expected invalid arguments, got: %#v", err.String())
	}
	if retcode != 6 {
		t.Errorf("Expected exit code 6, got: %d", retcode)
	}
}
