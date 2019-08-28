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

package help

import (
	"strings"

	"github.com/spf13/cobra"

	"k8s.io/kubectl/pkg/util/i18n"
	"k8s.io/kubectl/pkg/util/templates"
)

var helpLong = templates.LongDesc(i18n.T(`
	Help provides help for any command in the application.
	Simply type kubectl help [path to command] for full details.`))

// NewCmdHelp returns the help Cobra command
func NewCmdHelp() *cobra.Command {
	cmd := &cobra.Command{
		Use:                   "help [command] | STRING_TO_SEARCH",
		DisableFlagsInUseLine: true,
		Short:                 i18n.T("Help about any command"),
		Long:                  helpLong,

		Run: RunHelp,
	}

	return cmd
}

// RunHelp checks given arguments and executes command
func RunHelp(cmd *cobra.Command, args []string) {
	foundCmd, _, err := cmd.Root().Find(args)

	// NOTE(andreykurilin): actually, I did not find any cases when foundCmd can be nil,
	//   but let's make this check since it is included in original code of initHelpCmd
	//   from github.com/spf13/cobra
	if foundCmd == nil {
		cmd.Printf("Unknown help topic %#q.\n", args)
		cmd.Root().Usage()
	} else if err != nil {
		// print error message at first, since it can contain suggestions
		cmd.Println(err)

		argsString := strings.Join(args, " ")
		var matchedMsgIsPrinted = false
		for _, foundCmd := range foundCmd.Commands() {
			if strings.Contains(foundCmd.Short, argsString) {
				if !matchedMsgIsPrinted {
					cmd.Printf("Matchers of string '%s' in short descriptions of commands: \n", argsString)
					matchedMsgIsPrinted = true
				}
				cmd.Printf("  %-14s %s\n", foundCmd.Name(), foundCmd.Short)
			}
		}

		if !matchedMsgIsPrinted {
			// if nothing is found, just print usage
			cmd.Root().Usage()
		}
	} else {
		if len(args) == 0 {
			// help message for help command :)
			foundCmd = cmd
		}
		helpFunc := foundCmd.HelpFunc()
		helpFunc(foundCmd, args)
	}
}
