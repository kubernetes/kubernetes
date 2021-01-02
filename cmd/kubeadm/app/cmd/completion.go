/*
Copyright 2017 The Kubernetes Authors.

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

	"github.com/lithammer/dedent"
	"github.com/pkg/errors"
	"github.com/spf13/cobra"

	"k8s.io/component-base/cli/completion"
)

var (
	completionLong = dedent.Dedent(`
		Output shell completion code for the specified shell (bash or zsh).
		The shell code must be evaluated to provide interactive
		completion of kubeadm commands. This can be done by sourcing it from
		the .bash_profile.

		Note: this requires the bash-completion framework.

		To install it on Mac use homebrew:
		    $ brew install bash-completion
		Once installed, bash_completion must be evaluated. This can be done by adding the
		following line to the .bash_profile
		    $ source $(brew --prefix)/etc/bash_completion

		If bash-completion is not installed on Linux, please install the 'bash-completion' package
		via your distribution's package manager.

		Note for zsh users: [1] zsh completions are only supported in versions of zsh >= 5.2`)

	completionExample = dedent.Dedent(`
		# Install bash completion on a Mac using homebrew
		brew install bash-completion
		printf "\n# Bash completion support\nsource $(brew --prefix)/etc/bash_completion\n" >> $HOME/.bash_profile
		source $HOME/.bash_profile

		# Load the kubeadm completion code for bash into the current shell
		source <(kubeadm completion bash)

		# Write bash completion code to a file and source it from .bash_profile
		kubeadm completion bash > ~/.kube/kubeadm_completion.bash.inc
		printf "\n# Kubeadm shell completion\nsource '$HOME/.kube/kubeadm_completion.bash.inc'\n" >> $HOME/.bash_profile
		source $HOME/.bash_profile

		# Load the kubeadm completion code for zsh[1] into the current shell
		source <(kubeadm completion zsh)`)
)

// newCmdCompletion returns the "kubeadm completion" command
func newCmdCompletion(out io.Writer, boilerPlate string) *cobra.Command {
	cmd := &cobra.Command{
		Use:     "completion SHELL",
		Short:   "Output shell completion code for the specified shell (bash or zsh)",
		Long:    completionLong,
		Example: completionExample,
		RunE: func(cmd *cobra.Command, args []string) error {
			return RunCompletion(out, boilerPlate, cmd, args)
		},
		ValidArgs: completion.GetSupportedShells(),
	}

	return cmd
}

// RunCompletion checks given arguments and executes command
func RunCompletion(out io.Writer, boilerPlate string, cmd *cobra.Command, args []string) error {
	if length := len(args); length == 0 {
		return errors.New("shell not specified")
	} else if length > 1 {
		return errors.New("too many arguments. expected only the shell type")
	}

	return completion.RunCompletionForShell(out, boilerPlate, cmd, args[0])
}
