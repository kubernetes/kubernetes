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

package help

import (
	"k8s.io/code-generator/internal/codegen/execution"
	"strings"
)

// InvalidCommand is a command that prints an error message and the usage.
type InvalidCommand struct {
	Usage Command
}

func (i InvalidCommand) Matches(ex *execution.Vars) bool {
	// isn't used
	return false
}

func (i InvalidCommand) Run(ex *execution.Vars) {
	ex.Printf("%s: %s\n", i.OneLine(), strings.Join(ex.Args, " "))
	ex.Println()
	ex.Println(i.Usage.Help())
	ex.Exit(6)
}

func (i InvalidCommand) Name() string {
	return "invalid"
}

func (i InvalidCommand) OneLine() string {
	return "Invalid arguments given"
}

func (i InvalidCommand) Help() string {
	return i.OneLine()
}
