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

package completion

import (
	"io"

	"github.com/spf13/cobra"

	"k8s.io/component-base/cli/completion"
	cmdutil "k8s.io/kubectl/pkg/cmd/util"
	"k8s.io/kubectl/pkg/util/i18n"
	"k8s.io/kubectl/pkg/util/templates"
)

var (
	completionLong = templates.LongDesc(i18n.T(`
		Output shell completion code for the specified shell (bash or zsh).
		The shell code must be evaluated to provide interactive
		completion of kubectl commands.  This can be done by sourcing it from
		the .bash_profile.

		Detailed instructions on how to do this are available here:
		https://kubernetes.io/docs/tasks/tools/install-kubectl/#enabling-shell-autocompletion

		Note for zsh users: [1] zsh completions are only supported in versions of zsh >= 5.2`))

	completionExample = templates.Examples(i18n.T(`
		# Installing bash completion on macOS using homebrew
		## If running Bash 3.2 included with macOS
		    brew install bash-completion
		## or, if running Bash 4.1+
		    brew install bash-completion@2
		## If kubectl is installed via homebrew, this should start working immediately.
		## If you've installed via other means, you may need add the completion to your completion directory
		    kubectl completion bash > $(brew --prefix)/etc/bash_completion.d/kubectl


		# Installing bash completion on Linux
		## If bash-completion is not installed on Linux, please install the 'bash-completion' package
		## via your distribution's package manager.
		## Load the kubectl completion code for bash into the current shell
		    source <(kubectl completion bash)
		## Write bash completion code to a file and source it from .bash_profile
		    kubectl completion bash > ~/.kube/completion.bash.inc
		    printf "
		      # Kubectl shell completion
		      source '$HOME/.kube/completion.bash.inc'
		      " >> $HOME/.bash_profile
		    source $HOME/.bash_profile

		# Load the kubectl completion code for zsh[1] into the current shell
		    source <(kubectl completion zsh)
		# Set the kubectl completion code for zsh[1] to autoload on startup
		    kubectl completion zsh > "${fpath[1]}/_kubectl"`))
)

// NewCmdCompletion creates the `completion` command
func NewCmdCompletion(out io.Writer, boilerPlate string) *cobra.Command {
	cmd := &cobra.Command{
		Use:                   "completion SHELL",
		DisableFlagsInUseLine: true,
		Short:                 i18n.T("Output shell completion code for the specified shell (bash or zsh)"),
		Long:                  completionLong,
		Example:               completionExample,
		Run: func(cmd *cobra.Command, args []string) {
			err := RunCompletion(out, boilerPlate, cmd, args)
			cmdutil.CheckErr(err)
		},
		ValidArgs: completion.GetSupportedShells(),
	}

	return cmd
}

// RunCompletion checks given arguments and executes command
func RunCompletion(out io.Writer, boilerPlate string, cmd *cobra.Command, args []string) error {
	if len(args) == 0 {
		return cmdutil.UsageErrorf(cmd, "Shell not specified.")
	}
	if len(args) > 1 {
		return cmdutil.UsageErrorf(cmd, "Too many arguments. Expected only the shell type.")
	}
	return completion.RunCompletionForShell(out, boilerPlate, cmd, args[0])
}
