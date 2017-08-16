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

package sanity

import (
	"fmt"
	"os"
	"regexp"
	"strings"
	"unicode"

	"github.com/spf13/cobra"
	"github.com/spf13/pflag"
	"k8s.io/kubernetes/pkg/kubectl/cmd/templates"
)

type CmdCheck func(cmd *cobra.Command) []error
type GlobalCheck func() []error

var (
	AllCmdChecks = []CmdCheck{
		CheckLongDesc,
		CheckExamples,
		CheckFlags,
	}
	AllGlobalChecks = []GlobalCheck{
		CheckGlobalVarFlags,
	}
)

func RunGlobalChecks(globalChecks []GlobalCheck) []error {
	fmt.Fprint(os.Stdout, "---+ RUNNING GLOBAL CHECKS\n")
	errors := []error{}
	for _, check := range globalChecks {
		errors = append(errors, check()...)
	}
	return errors
}

func RunCmdChecks(cmd *cobra.Command, cmdChecks []CmdCheck, skipCmd []string) []error {
	cmdPath := cmd.CommandPath()

	for _, skipCmdPath := range skipCmd {
		if cmdPath == skipCmdPath {
			fmt.Fprintf(os.Stdout, "---+ skipping command %s\n", cmdPath)
			return []error{}
		}
	}

	errors := []error{}

	if cmd.HasSubCommands() {
		for _, subCmd := range cmd.Commands() {
			errors = append(errors, RunCmdChecks(subCmd, cmdChecks, skipCmd)...)
		}
	}

	fmt.Fprintf(os.Stdout, "---+ RUNNING COMMAND CHECKS on %q\n", cmdPath)

	for _, check := range cmdChecks {
		if err := check(cmd); err != nil && len(err) > 0 {
			errors = append(errors, err...)
		}
	}

	return errors
}

func CheckLongDesc(cmd *cobra.Command) []error {
	fmt.Fprint(os.Stdout, "   ↳ checking long description\n")
	cmdPath := cmd.CommandPath()
	long := cmd.Long
	if len(long) > 0 {
		if strings.Trim(long, " \t\n") != long {
			return []error{fmt.Errorf(`command %q: long description is not normalized, make sure you are calling templates.LongDesc (from pkg/cmd/templates) before assigning cmd.Long`, cmdPath)}
		}
	}
	return nil
}

type lineType int

const (
	LineInitial lineType = iota
	LineEmpty
	LineComment
	LineCommand
	LineMalformed
)

func CheckExamples(cmd *cobra.Command) []error {
	fmt.Fprint(os.Stdout, "   ↳ checking examples\n")

	examples := cmd.Example
	if len(examples) == 0 {
		return []error{}
	}

	cmdPath := cmd.CommandPath()
	errors := []error{}

	appendErr := func(exampleIndex int, msg string) {
		errors = append(errors, fmt.Errorf("command %q(subline: %d): %s", cmdPath, exampleIndex, msg))
	}

	seen := LineInitial
	for i, line := range strings.Split(examples, "\n") {
		if !isCorrectlyIndented(line) {
			appendErr(i, `examples are not normalized, make sure you are calling templates.Examples (from pkg/cmd/templates) before assigning cmd.Example`)
		}

		trimmed := strings.TrimSpace(line)

		if isEmpty(trimmed) {
			if seen == LineInitial {
				appendErr(i, "please don't start examples with a blank line")
			}
			if seen == LineEmpty {
				appendErr(i, "please leave only a single blank line between examples")
			}
			seen = LineEmpty
			continue
		}

		if !(isComment(trimmed) || isCommand(trimmed)) {
			appendErr(i, "examples should either be comments (#) or commands ($), please start example line with the correct character for it's type.")
			seen = LineMalformed
		}

		if isComment(trimmed) {
			if !strings.HasPrefix(trimmed, "# ") {
				appendErr(i, "please add a space after the hash character for Example comments")
			} else {
				// Allow multi-line comments, enforce capitalization on initial line only.
				if seen != LineComment && thirdCharacterIsLower(trimmed) {
					appendErr(i, "please capitalize Example comments")
				}
			}

			if !strings.HasSuffix(trimmed, ".") {
				appendErr(i, "please terminate Example comments with a period")
			}
			seen = LineComment
		} else if isCommand(trimmed) {
			if !strings.HasPrefix(trimmed, "$ ") {
				appendErr(i, "please add a space after the dollar sign for Example commands")
			}
			seen = LineCommand
		}
	}
	return errors
}

func isCorrectlyIndented(line string) bool {
	return strings.HasPrefix(line, templates.Indentation)
}

func isEmpty(line string) bool {
	return line == ""
}

func isComment(line string) bool {
	return strings.HasPrefix(line, "#")
}

func isCommand(line string) bool {
	return strings.HasPrefix(line, "$")
}

// thirdCharacterIsLower returns true if third character of s is a capital.
func thirdCharacterIsLower(s string) bool {
	if len(s) < 3 {
		return false
	}
	runes := []rune(s)
	if !unicode.IsLetter(runes[2]) {
		return false
	}
	return unicode.IsLower(runes[2])
}

func CheckFlags(cmd *cobra.Command) []error {
	allFlagsSlice := []*pflag.Flag{}

	cmd.Flags().VisitAll(func(f *pflag.Flag) {
		allFlagsSlice = append(allFlagsSlice, f)
	})
	cmd.PersistentFlags().VisitAll(func(f *pflag.Flag) {
		allFlagsSlice = append(allFlagsSlice, f)
	})

	fmt.Fprintf(os.Stdout, "   ↳ checking %d flags\n", len(allFlagsSlice))

	errors := []error{}

	// check flags long names
	regex, err := regexp.Compile(`^[a-z]+[a-z\-]*$`)
	if err != nil {
		errors = append(errors, fmt.Errorf("command %q: unable to compile regex to check flags", cmd.CommandPath()))
		return errors
	}
	for _, flag := range allFlagsSlice {
		name := flag.Name
		if !regex.MatchString(name) {
			errors = append(errors, fmt.Errorf("command %q: flag name %q is invalid, long form of flag names can only contain lowercase characters or dash (must match %v)", cmd.CommandPath(), name, regex))
		}
	}

	return errors
}

func CheckGlobalVarFlags() []error {
	fmt.Fprint(os.Stdout, "   ↳ checking flags from global vars\n")
	errors := []error{}
	pflag.CommandLine.VisitAll(func(f *pflag.Flag) {
		errors = append(errors, fmt.Errorf("flag %q is invalid, please don't register any flag under the global variable \"CommandLine\"", f.Name))
	})
	return errors
}
