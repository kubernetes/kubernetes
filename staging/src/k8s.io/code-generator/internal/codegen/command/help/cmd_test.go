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
	"k8s.io/code-generator/internal/codegen/command"
	"k8s.io/code-generator/internal/codegen/command/help"
	"k8s.io/code-generator/internal/codegen/execution"
	"testing"
)

func TestHelpCommand(t *testing.T) {
	t.Parallel()
	c := help.Command{
		Others: []command.Usage{
			testUsage{"foo", "foo one line", "foo help"},
			testUsage{"bar", "bar one line", "bar help"},
		},
	}
	var err bytes.Buffer
	c.Run(execution.New(func(v *execution.Vars) {
		v.Out = &err
		v.Args = []string{"help"}
	}))

	want := `Usage: code-generator [command] [options]

Command:
  help    Print this message. You can specify a command to get help for that command.
  foo     foo one line
  bar     bar one line

`
	if err.String() != want {
		t.Errorf("Output missmaches:\nwant %q,\n got %q", want, err.String())
	}
}

type testUsage struct {
	name string
	one  string
	help string
}

func (t testUsage) Name() string {
	return t.name
}

func (t testUsage) OneLine() string {
	return t.one
}

func (t testUsage) Help() string {
	return t.help
}
