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
	"fmt"
	"k8s.io/code-generator/internal/codegen/command"
	"k8s.io/code-generator/internal/codegen/execution"
	"strings"
)

// Command prints the usage information on the standard output.
type Command struct {
	Others []command.Usage
}

func (c Command) Name() string {
	return "help"
}

func (c Command) OneLine() string {
	return "Print this message. You can specify a command to get help for that command."
}

func (c Command) Help() string {
	var sb strings.Builder
	_, _ = sb.WriteString(`Usage: code-generator [command] [options]

Command:
`)
	cmds := append([]command.Usage{c}, c.Others...)
	longest := 0
	type mesg struct {
		name string
		desc string
	}
	var mesgs []mesg
	for _, usage := range cmds {
		currLen := len(usage.Name())
		if currLen > longest {
			longest = currLen
		}
		mesgs = append(mesgs, mesg{name: usage.Name(), desc: usage.OneLine()})
	}
	longest += 3
	for _, m := range mesgs {
		_, _ = sb.WriteString(fmt.Sprintf("  %s %s\n",
			pad(m.name, longest), m.desc))
	}
	return sb.String()
}

func pad(s string, l int) string {
	pl := l - len(s)
	if pl < 0 {
		pl = 0
	}
	return s + strings.Repeat(" ", pl)
}

func (c Command) Matches(ex *execution.Vars) bool {
	if len(ex.Args) >= 1 && ex.Args[0] == "help" {
		return true
	}
	for _, arg := range ex.Args {
		if arg == "--help" || arg == "-h" {
			return true
		}
	}
	return false
}

func (c Command) Run(ex *execution.Vars) {
	args := stripHelp(ex.Args)
	if len(args) > 0 {
		cmdName := args[0]
		cmds := append([]command.Usage{c}, c.Others...)
		for _, c := range cmds {
			if c.Name() == cmdName {
				ex.Println(c.Help())
				return
			}
		}
		i := InvalidCommand{}
		i.Run(ex)
		return
	}
	ex.Println(c.Help())
}

func stripHelp(args []string) []string {
	filtered := make([]string, 0, len(args))
	for _, arg := range args {
		if arg == "help" || arg == "--help" || arg == "-h" {
			continue
		}
		filtered = append(filtered, arg)
	}
	return filtered
}
