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

package help_test

import (
	"bytes"
	"k8s.io/code-generator/internal/codegen/command/help"
	"k8s.io/code-generator/internal/codegen/execution"
	"math"
	"testing"
)

func TestInvalidCommand(t *testing.T) {
	t.Parallel()
	i := help.InvalidCommand{
		Usage: help.Command{},
	}
	var err bytes.Buffer
	retcode := math.MinInt32
	v := execution.New(func(v *execution.Vars) {
		v.Args = []string{"invalid-cmd"}
		v.Out = &err
		v.Exit = func(code int) {
			retcode = code
		}
	})
	if i.Matches(v) {
		t.Error("Expected no match")
	}
	if i.Name() != "invalid" {
		t.Errorf("Expected name invalid, got %q", i.Name())
	}
	if i.OneLine() != "Invalid arguments given" {
		t.Errorf("Expected one line message, got %q", i.OneLine())
	}
	if i.Help() != "Invalid arguments given" {
		t.Errorf("Expected help message, got %q", i.Help())
	}
	i.Run(v)
	if retcode != 6 {
		t.Errorf("Expected exit code 6, got %d", retcode)
	}
	want := `Invalid arguments given: invalid-cmd

Usage: code-generator [command] [options]

Command:
  help    Print this message. You can specify a command to get help for that command.

`
	if err.String() != want {
		t.Errorf("Output missmaches:\nwant %q,\n got %q", want, err.String())
	}
}
