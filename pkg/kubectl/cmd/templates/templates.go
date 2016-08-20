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

package templates

import "strings"

func MainHelpTemplate() string {
	return decorate(mainHelpTemplate, false)
}

func MainUsageTemplate() string {
	return decorate(mainUsageTemplate, true) + "\n"
}

func OptionsHelpTemplate() string {
	return decorate(optionsHelpTemplate, false)
}

func OptionsUsageTemplate() string {
	return decorate(optionsUsageTemplate, false)
}

func decorate(template string, trim bool) string {
	if trim && len(strings.Trim(template, " ")) > 0 {
		template = strings.Trim(template, "\n")
	}
	return template
}

const (
	vars = `{{$isRootCmd := isRootCmd .}}` +
		`{{$rootCmd := rootCmd .}}` +
		`{{$visibleFlags := visibleFlags (flagsNotIntersected .LocalFlags .PersistentFlags)}}` +
		`{{$explicitlyExposedFlags := exposed .}}` +
		`{{$optionsCmdFor := optionsCmdFor .}}` +
		`{{$usageLine := usageLine .}}`

	mainHelpTemplate = `{{with or .Long .Short }}{{. | trim}}{{end}}{{if or .Runnable .HasSubCommands}}{{.UsageString}}{{end}}`

	mainUsageTemplate = vars +
		// ALIASES
		`{{if gt .Aliases 0}}

Aliases:
{{.NameAndAliases}}{{end}}` +

		// EXAMPLES
		`{{if .HasExample}}

Examples:
{{ indentLines (.Example | trimLeft) 2 }}{{end}}` +

		// SUBCOMMANDS
		`{{ if .HasAvailableSubCommands}}
{{range cmdGroups . .Commands}}
{{.Message}}
{{range .Commands}}{{if .Runnable}}  {{rpad .Name .NamePadding }} {{.Short}}
{{end}}{{end}}{{end}}{{end}}` +

		// VISIBLE FLAGS
		`{{ if or $visibleFlags.HasFlags $explicitlyExposedFlags.HasFlags}}

Options:
{{ if $visibleFlags.HasFlags}}{{flagsUsages $visibleFlags}}{{end}}{{ if $explicitlyExposedFlags.HasFlags}}{{flagsUsages $explicitlyExposedFlags}}{{end}}{{end}}` +

		// USAGE LINE
		`{{if and .Runnable (ne .UseLine "") (ne .UseLine $rootCmd)}}
Usage:
  {{$usageLine}}
{{end}}` +

		// TIPS: --help
		`{{ if .HasSubCommands }}
Use "{{$rootCmd}} <command> --help" for more information about a given command.{{end}}` +

		// TIPS: global options
		`{{ if $optionsCmdFor}}
Use "{{$optionsCmdFor}}" for a list of global command-line options (applies to all commands).{{end}}`

	optionsHelpTemplate = ``

	optionsUsageTemplate = `{{ if .HasInheritedFlags}}The following options can be passed to any command:

{{flagsUsages .InheritedFlags}}{{end}}`
)
