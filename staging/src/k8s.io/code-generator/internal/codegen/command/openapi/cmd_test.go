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

package openapi_test

import (
	"bytes"
	goflag "flag"
	"math"
	"testing"

	"k8s.io/code-generator/internal/codegen/command/openapi"
	"k8s.io/code-generator/internal/codegen/execution"
	pkgopenapi "k8s.io/code-generator/pkg/codegen/openapi"
)

func TestCommandMatches(t *testing.T) {
	t.Parallel()

	cmd := openapi.Command{Gen: &testGen{}}
	ex := execution.New(func(v *execution.Vars) {
		v.Args = []string{cmd.Name()}
	})
	if !cmd.Matches(ex) {
		t.Errorf("expected command to match")
	}
}

func TestCommandRun(t *testing.T) {
	t.Parallel()

	gen := &testGen{}
	cmd := openapi.Command{
		Gen:     gen,
		FlagSet: goflag.NewFlagSet("test", goflag.ContinueOnError),
	}
	ex := execution.New(func(v *execution.Vars) {
		v.Args = []string{
			cmd.Name(),
			"--input-dir", "foo",
		}
	})
	cmd.Run(ex)
	if len(gen.calls) != 1 {
		t.Errorf("expected gen to be called once, got %d", len(gen.calls))
	}
	call := gen.calls[0]
	if call.InputDir != "foo" {
		t.Errorf("expected input package to be foo, got %s", call.InputDir)
	}
}

func TestCommandRunInvalidArgs(t *testing.T) {
	t.Parallel()
	var errstream bytes.Buffer
	retcode := math.MinInt32
	gen := &testGen{}
	cmd := openapi.Command{
		Gen:     gen,
		FlagSet: goflag.NewFlagSet("test", goflag.ContinueOnError),
	}
	ex := execution.New(func(v *execution.Vars) {
		v.Args = []string{cmd.Name(), "--input-foo", "foo"}
		v.Out = &errstream
		v.Exit = func(code int) {
			retcode = code
		}
	})
	cmd.Run(ex)
	if len(gen.calls) != 0 {
		t.Errorf("expected gen to be called zero times, got %d", len(gen.calls))
	}
	if retcode != 12 {
		t.Errorf("expected exit code 12, got %d", retcode)
	}
	if errstream.String() == "" {
		t.Errorf("expected error message to be printed")
	}
}

type testGen struct {
	calls []pkgopenapi.Args
}

func (t *testGen) Generate(args *pkgopenapi.Args) error {
	if t.calls == nil {
		t.calls = make([]pkgopenapi.Args, 0, 1)
	}
	t.calls = append(t.calls, *args)
	return nil
}
