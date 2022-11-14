/*
Copyright 2022 The Kubernetes Authors.

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
	"strings"

	"github.com/spf13/cobra"

	"k8s.io/klog/v2"
)

// AliasHandler is responsible for injecting aliases for commands and
// arguments based on user's configuration.
type AliasHandler interface {
	InjectAll(rootCmd *cobra.Command)
}

type alias struct {
	name string
	args []string
}

// DefaultAliasHandler implements AliasHandler
type DefaultAliasHandler struct {
	aliasesMap map[string]alias
}

// NewDefaultAliasHandler instantiates the DefaultAliasHandler by reading the
// kuberc file.
func NewDefaultAliasHandler() *DefaultAliasHandler {
	// TODO: read aliases from .kuberc file
	// TODO: handle sub-commands, like create deployment, for example
	return &DefaultAliasHandler{
		aliasesMap: map[string]alias{
			"delete": {"delete", []string{"--dry-run=client"}},
			"events": {"ev", []string{"--types Warning", "--types Normal"}},
			// FIXME: this doesn't work correctly, since alias is created at the
			// end of the chain, and it should rather be at the top
			"create deployment": {"cd", []string{}},
		},
	}
}

func (h *DefaultAliasHandler) InjectAll(rootCmd *cobra.Command) {
	// FIXME: questions to answer:
	// 1. currently --help will show the aliases - we could either hide aliases
	//    or let it be as is, but the help will differ for users and won't
	//    show the additional arguments, which might be confusing
	// 2. what about completions?
	for command, alias := range h.aliasesMap {
		commands := strings.Split(command, " ")
		cmd, flags, err := rootCmd.Find(commands)
		if err != nil {
			klog.Warningf("Command %q not found to set alias %q: %v", command, alias.name, flags)
			continue
		}
		// do not allow shadowing built-ins
		if _, _, err := rootCmd.Find([]string{alias.name}); err == nil {
			klog.Warningf("Setting alias %q to a built-in command is not supported", alias.name)
			continue
		}
		// register alias
		cmd.Aliases = append(cmd.Aliases, alias.name)
		// inject alias flags
		cmd.Flags().Parse(alias.args)
	}
}
