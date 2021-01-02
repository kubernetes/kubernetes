/*
Copyright 2021 The Kubernetes Authors.

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

package completion

import (
	"fmt"
	"io"

	"github.com/spf13/cobra"
)

var (
	completionShells = map[string]func(out io.Writer, cmd *cobra.Command) error{
		"bash": func(out io.Writer, cmd *cobra.Command) error {
			return cmd.GenBashCompletion(out)
		},
		"zsh": func(out io.Writer, cmd *cobra.Command) error {
			return cmd.GenBashCompletion(out)
		},
	}
)

const defaultBoilerPlate = `
# Copyright 2021 The Kubernetes Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
`

// GetSupportedShells returns a list of shells that support completion.
func GetSupportedShells() []string {
	shells := []string{}
	for s := range completionShells {
		shells = append(shells, s)
	}
	return shells
}

// RunCompletionForShell runs the completion generator for the specified shell.
func RunCompletionForShell(out io.Writer, boilerPlate string, cmd *cobra.Command, shell string) error {
	run, found := completionShells[shell]
	if !found {
		return fmt.Errorf("unsupported shell type %q", shell)
	}

	if len(boilerPlate) == 0 {
		boilerPlate = defaultBoilerPlate
	}

	if _, err := out.Write([]byte(boilerPlate)); err != nil {
		return err
	}

	return run(out, cmd.Parent())
}
