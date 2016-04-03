// Copyright 2015 The appc Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package main

import (
	"flag"
	"fmt"
	"strings"
	"text/template"

	"github.com/appc/spec/schema"
)

var (
	cmdHelp = &Command{
		Name:        "help",
		Summary:     "Show a list of commands or help for one command",
		Usage:       "[COMMAND]",
		Description: "Show a list of commands or detailed help for one command",
		Run:         runHelp,
	}

	globalUsageTemplate  *template.Template
	commandUsageTemplate *template.Template
	templFuncs           = template.FuncMap{
		"descToLines": func(s string) []string {
			// trim leading/trailing whitespace and split into slice of lines
			return strings.Split(strings.Trim(s, "\n\t "), "\n")
		},
		"printOption": func(name, defvalue, usage string) string {
			prefix := "--"
			if len(name) == 1 {
				prefix = "-"
			}
			return fmt.Sprintf("\t%s%s=%s\t%s", prefix, name, defvalue, usage)
		},
	}
)

func init() {
	globalUsageTemplate = template.Must(template.New("global_usage").Funcs(templFuncs).Parse(`
NAME:
{{printf "\t%s - %s" .Executable .Description}}

USAGE:
{{printf "\t%s" .Executable}} [global options] <command> [command options] [arguments...]

VERSION:
{{printf "\t%s" .Version}}

COMMANDS:{{range .Commands}}
{{printf "\t%s\t%s" .Name .Summary}}{{end}}

GLOBAL OPTIONS:{{range .Flags}}
{{printOption .Name .DefValue .Usage}}{{end}}

Run "{{.Executable}} help <command>" for more details on a specific command.
`[1:]))
	commandUsageTemplate = template.Must(template.New("command_usage").Funcs(templFuncs).Parse(`
NAME:
{{printf "\t%s - %s" .Cmd.Name .Cmd.Summary}}

USAGE:
{{printf "\t%s %s %s" .Executable .Cmd.Name .Cmd.Usage}}

DESCRIPTION:
{{range $line := descToLines .Cmd.Description}}{{printf "\t%s" $line}}
{{end}}
{{if .CmdFlags}}OPTIONS:{{range .CmdFlags}}
{{printOption .Name .DefValue .Usage}}{{end}}

{{end}}For help on global options run "{{.Executable}} help"
`[1:]))
}

func runHelp(args []string) (exit int) {
	if len(args) < 1 {
		printGlobalUsage()
		return
	}

	var cmd *Command

	for _, c := range commands {
		if c.Name == args[0] {
			cmd = c
			break
		}
	}

	if cmd == nil {
		stderr("Unrecognized command: %s", args[0])
		return 1
	}

	printCommandUsage(cmd)
	return
}

func printGlobalUsage() {
	globalUsageTemplate.Execute(out, struct {
		Executable  string
		Commands    []*Command
		Flags       []*flag.Flag
		Description string
		Version     string
	}{
		cliName,
		commands,
		getAllFlags(),
		cliDescription,
		schema.AppContainerVersion.String(),
	})
	out.Flush()
}

func printCommandUsage(cmd *Command) {
	commandUsageTemplate.Execute(out, struct {
		Executable string
		Cmd        *Command
		CmdFlags   []*flag.Flag
	}{
		cliName,
		cmd,
		getFlags(&cmd.Flags),
	})
	out.Flush()
}
