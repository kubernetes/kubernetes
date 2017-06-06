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

func CheckExamples(cmd *cobra.Command) []error {
	fmt.Fprint(os.Stdout, "   ↳ checking examples\n")
	cmdPath := cmd.CommandPath()
	examples := cmd.Example
	errors := []error{}
	if len(examples) > 0 {
		for _, line := range strings.Split(examples, "\n") {
			if !strings.HasPrefix(line, templates.Indentation) {
				errors = append(errors, fmt.Errorf(`command %q: examples are not normalized, make sure you are calling templates.Examples (from pkg/cmd/templates) before assigning cmd.Example`, cmdPath))
			}
			if trimmed := strings.TrimSpace(line); strings.HasPrefix(trimmed, "//") {
				errors = append(errors, fmt.Errorf(`command %q: we use # to start comments in examples instead of //`, cmdPath))
			}
		}
	}
	return errors
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
