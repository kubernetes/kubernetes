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

package codegen

import (
	"k8s.io/code-generator/internal/codegen/command"
	"k8s.io/code-generator/internal/codegen/execution"
)

// Command is a command that can be run by code-gen.
type Command interface {
	// Matches returns true if the command matches the execution context.
	Matches(ex *execution.Vars) bool
	// Run runs the command.
	Run(ex *execution.Vars)
	command.Usage
}
