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

import (
	"strings"
	"unicode"
)

const (
	// SectionVars is the help template section that declares variables to be used in the template.
	SectionVars = `{{$isRootCmd := isRootCmd .}}` +
		`{{$rootCmd := rootCmd .}}` +
		`{{$visibleFlags := visibleFlags (flagsNotIntersected .LocalFlags .PersistentFlags)}}` +
		`{{$explicitlyExposedFlags := exposed .}}` +
		`{{$optionsCmdFor := optionsCmdFor .}}` +
		`{{$usageLine := usageLine .}}`

	// SectionAliases is the help template section that displays command aliases.
	SectionAliases = `{{if gt .Aliases 0}}Aliases:
{{.NameAndAliases}}

{{end}}`

	// SectionExamples is the help template section that displays command examples.
	SectionExamples = `{{if .HasExample}}Examples:
{{trimRight .Example}}

{{end}}`

	// SectionSubcommands is the help template section that displays the command's subcommands.
	SectionSubcommands = `{{if .HasAvailableSubCommands}}{{cmdGroupsString .}}

{{end}}`

	// SectionFlags is the help template section that displays the command's flags.
	SectionFlags = `{{ if or $visibleFlags.HasFlags $explicitlyExposedFlags.HasFlags}}Options:
{{ if $visibleFlags.HasFlags}}{{trimRight (flagsUsages $visibleFlags)}}{{end}}{{ if $explicitlyExposedFlags.HasFlags}}{{ if $visibleFlags.HasFlags}}
{{end}}{{trimRight (flagsUsages $explicitlyExposedFlags)}}{{end}}

{{end}}`

	// SectionUsage is the help template section that displays the command's usage.
	SectionUsage = `{{if and .Runnable (ne .UseLine "") (ne .UseLine $rootCmd)}}Usage:
  {{$usageLine}}

{{end}}`

	// SectionTipsHelp is the help template section that displays the '--help' hint.
	SectionTipsHelp = `{{if .HasSubCommands}}Use "{{$rootCmd}} <command> --help" for more information about a given command.
{{end}}`

	// SectionTipsGlobalOptions is the help template section that displays the 'options' hint for displaying global flags.
	SectionTipsGlobalOptions = `{{if $optionsCmdFor}}Use "{{$optionsCmdFor}}" for a list of global command-line options (applies to all commands).
{{end}}`
)

// MainHelpTemplate if the template for 'help' used by most commands.
func MainHelpTemplate() string {
	return `{{with or .Long .Short }}{{. | trim}}{{end}}{{if or .Runnable .HasSubCommands}}{{.UsageString}}{{end}}`
}

// MainUsageTemplate if the template for 'usage' used by most commands.
func MainUsageTemplate() string {
	sections := []string{
		"\n\n",
		SectionVars,
		SectionAliases,
		SectionExamples,
		SectionSubcommands,
		SectionFlags,
		SectionUsage,
		SectionTipsHelp,
		SectionTipsGlobalOptions,
	}
	return strings.TrimRightFunc(strings.Join(sections, ""), unicode.IsSpace)
}

// OptionsHelpTemplate if the template for 'help' used by the 'options' command.
func OptionsHelpTemplate() string {
	return ""
}

// OptionsUsageTemplate if the template for 'usage' used by the 'options' command.
func OptionsUsageTemplate() string {
	return `{{ if .HasInheritedFlags}}The following options can be passed to any command:

{{flagsUsages .InheritedFlags}}{{end}}`
}
