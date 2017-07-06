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

package cmd

import (
	"io"

	"k8s.io/kubernetes/pkg/kubectl/cmd/templates"
	"k8s.io/kubernetes/pkg/util/i18n"

	"github.com/spf13/cobra"
)

var (
	optionsExample = templates.Examples(i18n.T(`
		# Print flags inherited by all commands
		kubectl options`))
)

// NewCmdOptions implements the options command
func NewCmdOptions(out io.Writer) *cobra.Command {
	cmd := &cobra.Command{
		Use:     "options",
		Short:   i18n.T("Print the list of flags inherited by all commands"),
		Long:    "Print the list of flags inherited by all commands",
		Example: optionsExample,
		Run: func(cmd *cobra.Command, args []string) {
			cmd.Usage()
		},
	}

	// The `options` command needs write its output to the `out` stream
	// (typically stdout). Without calling SetOutput here, the Usage()
	// function call will fall back to stderr.
	//
	// See https://github.com/kubernetes/kubernetes/pull/46394 for details.
	cmd.SetOutput(out)

	templates.UseOptionsTemplates(cmd)
	return cmd
}
