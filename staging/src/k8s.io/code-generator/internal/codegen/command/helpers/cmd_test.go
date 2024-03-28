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

package helpers_test

import (
	"bytes"
	goflag "flag"
	"math"
	"os"
	"path"
	"strings"
	"testing"

	"k8s.io/code-generator/internal/codegen/command/helpers"
	"k8s.io/code-generator/internal/codegen/execution"
	pkghelpers "k8s.io/code-generator/pkg/codegen/helpers"
)

func TestCommandMatches(t *testing.T) {
	t.Parallel()

	cmd := helpers.Command{Gen: &testGen{}}
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
	cmd := helpers.Command{
		Gen:     gen,
		FlagSet: goflag.NewFlagSet("test", goflag.ContinueOnError),
	}
	ob := t.TempDir()
	inputDir := path.Join(ob, "foo")
	_ = os.MkdirAll(inputDir, 0x755)
	ex := execution.New(func(v *execution.Vars) {
		v.Args = []string{
			cmd.Name(),
			"--extra-peer-dir", "bar",
			inputDir,
		}
	})
	cmd.Run(ex)
	if len(gen.calls) != 1 {
		t.Errorf("expected gen to be called once, got %d", len(gen.calls))
	}
	call := gen.calls[0]
	if call.InputDir != inputDir {
		t.Errorf("expected input package to be %s, got %s",
			inputDir, call.InputDir)
	}
	if !strings.Contains(call.Boilerplate, "boilerplate.go.txt") {
		t.Errorf("expected boilerplate to point to boilerplate.go.txt, got %s",
			call.Boilerplate)
	}
	if len(call.ExtraPeerDirs) != 1 || call.ExtraPeerDirs[0] != "bar" {
		t.Errorf("expected extra peer dirs to be [bar], got %v",
			call.ExtraPeerDirs)

	}
}

func TestCommandRunInvalidArgs(t *testing.T) {
	t.Parallel()
	var errstream bytes.Buffer
	retcode := math.MinInt32
	gen := &testGen{}
	cmd := helpers.Command{
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
	if retcode != 10 {
		t.Errorf("expected exit code 10, got %d", retcode)
	}
	if errstream.String() == "" {
		t.Errorf("expected error message to be printed")
	}
}

type testGen struct {
	calls []pkghelpers.Args
}

func (t *testGen) Generate(args *pkghelpers.Args) error {
	if t.calls == nil {
		t.calls = make([]pkghelpers.Args, 0, 1)
	}
	t.calls = append(t.calls, *args)
	return nil
}
