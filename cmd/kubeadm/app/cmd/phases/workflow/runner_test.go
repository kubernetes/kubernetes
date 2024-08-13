/*
Copyright 2018 The Kubernetes Authors.

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

package workflow

import (
	"fmt"
	"reflect"
	"strings"
	"testing"

	"github.com/pkg/errors"
	"github.com/spf13/cobra"
	"github.com/spf13/pflag"
)

func phaseBuilder(name string, phases ...Phase) Phase {
	return Phase{
		Name:   name,
		Short:  fmt.Sprintf("long description for %s ...", name),
		Phases: phases,
	}
}

func TestComputePhaseRunFlags(t *testing.T) {

	var usecases = []struct {
		name          string
		options       RunnerOptions
		expected      map[string]bool
		expectedError bool
	}{
		{
			name:     "no options > all phases",
			options:  RunnerOptions{},
			expected: map[string]bool{"foo": true, "foo/bar": true, "foo/baz": true, "qux": true},
		},
		{
			name:     "options can filter phases",
			options:  RunnerOptions{FilterPhases: []string{"foo/baz", "qux"}},
			expected: map[string]bool{"foo": false, "foo/bar": false, "foo/baz": true, "qux": true},
		},
		{
			name:     "options can filter phases - hierarchy is considered",
			options:  RunnerOptions{FilterPhases: []string{"foo"}},
			expected: map[string]bool{"foo": true, "foo/bar": true, "foo/baz": true, "qux": false},
		},
		{
			name:     "options can skip phases",
			options:  RunnerOptions{SkipPhases: []string{"foo/bar", "qux"}},
			expected: map[string]bool{"foo": true, "foo/bar": false, "foo/baz": true, "qux": false},
		},
		{
			name:     "options can skip phases - hierarchy is considered",
			options:  RunnerOptions{SkipPhases: []string{"foo"}},
			expected: map[string]bool{"foo": false, "foo/bar": false, "foo/baz": false, "qux": true},
		},
		{
			name: "skip options have higher precedence than filter options",
			options: RunnerOptions{
				FilterPhases: []string{"foo"},     //  "foo", "foo/bar", "foo/baz" true
				SkipPhases:   []string{"foo/bar"}, //  "foo/bar" false
			},
			expected: map[string]bool{"foo": true, "foo/bar": false, "foo/baz": true, "qux": false},
		},
		{
			name:          "invalid filter option",
			options:       RunnerOptions{FilterPhases: []string{"invalid"}},
			expectedError: true,
		},
		{
			name:          "invalid skip option",
			options:       RunnerOptions{SkipPhases: []string{"invalid"}},
			expectedError: true,
		},
	}
	for _, u := range usecases {
		t.Run(u.name, func(t *testing.T) {
			var w = Runner{
				Phases: []Phase{
					phaseBuilder("foo",
						phaseBuilder("bar"),
						phaseBuilder("baz"),
					),
					phaseBuilder("qux"),
				},
			}

			w.prepareForExecution()
			w.Options = u.options
			actual, err := w.computePhaseRunFlags()
			if (err != nil) != u.expectedError {
				t.Errorf("Unexpected error: %v", err)
			}
			if err != nil {
				return
			}
			if !reflect.DeepEqual(actual, u.expected) {
				t.Errorf("\nactual:\n\t%v\nexpected:\n\t%v\n", actual, u.expected)
			}
		})
	}
}

func phaseBuilder1(name string, runIf func(data RunData) (bool, error), phases ...Phase) Phase {
	return Phase{
		Name:   name,
		Short:  fmt.Sprintf("long description for %s ...", name),
		Phases: phases,
		Run:    runBuilder(name),
		RunIf:  runIf,
	}
}

var callstack []string

func runBuilder(name string) func(data RunData) error {
	return func(data RunData) error {
		callstack = append(callstack, name)
		return nil
	}
}

func runConditionTrue(data RunData) (bool, error) {
	return true, nil
}

func runConditionFalse(data RunData) (bool, error) {
	return false, nil
}

func TestRunOrderAndConditions(t *testing.T) {
	var w = Runner{
		Phases: []Phase{
			phaseBuilder1("foo", nil,
				phaseBuilder1("bar", runConditionTrue),
				phaseBuilder1("baz", runConditionFalse),
			),
			phaseBuilder1("qux", runConditionTrue),
		},
	}

	var usecases = []struct {
		name          string
		options       RunnerOptions
		expectedOrder []string
	}{
		{
			name:          "Run respect runCondition",
			expectedOrder: []string{"foo", "bar", "qux"},
		},
		{
			name:          "Run takes options into account",
			options:       RunnerOptions{FilterPhases: []string{"foo"}, SkipPhases: []string{"foo/baz"}},
			expectedOrder: []string{"foo", "bar"},
		},
	}
	for _, u := range usecases {
		t.Run(u.name, func(t *testing.T) {
			callstack = []string{}
			w.Options = u.options
			err := w.Run([]string{})
			if err != nil {
				t.Errorf("Unexpected error: %v", err)
			}
			if !reflect.DeepEqual(callstack, u.expectedOrder) {
				t.Errorf("\ncallstack:\n\t%v\nexpected:\n\t%v\n", callstack, u.expectedOrder)
			}
		})
	}
}

func phaseBuilder2(name string, runIf func(data RunData) (bool, error), run func(data RunData) error, phases ...Phase) Phase {
	return Phase{
		Name:   name,
		Short:  fmt.Sprintf("long description for %s ...", name),
		Phases: phases,
		Run:    run,
		RunIf:  runIf,
	}
}

func runPass(data RunData) error {
	return nil
}

func runFails(data RunData) error {
	return errors.New("run fails")
}

func runConditionPass(data RunData) (bool, error) {
	return true, nil
}

func runConditionFails(data RunData) (bool, error) {
	return false, errors.New("run condition fails")
}

func TestRunHandleErrors(t *testing.T) {
	var w = Runner{
		Phases: []Phase{
			phaseBuilder2("foo", runConditionPass, runPass),
			phaseBuilder2("bar", runConditionPass, runFails),
			phaseBuilder2("baz", runConditionFails, runPass),
		},
	}

	var usecases = []struct {
		name          string
		options       RunnerOptions
		expectedError bool
	}{
		{
			name:    "no errors",
			options: RunnerOptions{FilterPhases: []string{"foo"}},
		},
		{
			name:          "run fails",
			options:       RunnerOptions{FilterPhases: []string{"bar"}},
			expectedError: true,
		},
		{
			name:          "run condition fails",
			options:       RunnerOptions{FilterPhases: []string{"baz"}},
			expectedError: true,
		},
	}
	for _, u := range usecases {
		t.Run(u.name, func(t *testing.T) {
			w.Options = u.options
			err := w.Run([]string{})
			if (err != nil) != u.expectedError {
				t.Errorf("Unexpected error: %v", err)
			}
		})
	}
}

func phaseBuilder3(name string, hidden bool, phases ...Phase) Phase {
	return Phase{
		Name:   name,
		Short:  fmt.Sprintf("long description for %s ...", name),
		Phases: phases,
		Hidden: hidden,
	}
}

func TestHelp(t *testing.T) {
	var w = Runner{
		Phases: []Phase{
			phaseBuilder3("foo", false,
				phaseBuilder3("bar [arg]", false),
				phaseBuilder3("baz", true),
			),
			phaseBuilder3("qux", false),
		},
	}

	expected := "The \"myCommand\" command executes the following phases:\n" +
		"```\n" +
		"foo   long description for foo ...\n" +
		"  /bar  long description for bar [arg] ...\n" +
		"qux   long description for qux ...\n" +
		"```"

	actual := w.Help("myCommand")
	if !reflect.DeepEqual(actual, expected) {
		t.Errorf("\nactual:\n\t%v\nexpected:\n\t%v\n", actual, expected)
	}
}

func phaseBuilder4(name string, cmdFlags []string, phases ...Phase) Phase {
	return Phase{
		Name:         name,
		Phases:       phases,
		InheritFlags: cmdFlags,
	}
}

func phaseBuilder5(name string, flags *pflag.FlagSet) Phase {
	return Phase{
		Name:       name,
		LocalFlags: flags,
	}
}

type argTest struct {
	args cobra.PositionalArgs
	pass []string
	fail []string
}

func phaseBuilder6(name string, args cobra.PositionalArgs, phases ...Phase) Phase {
	return Phase{
		Name:          name,
		Short:         fmt.Sprintf("long description for %s ...", name),
		Phases:        phases,
		ArgsValidator: args,
	}
}

// customArgs is a custom cobra.PositionArgs function
func customArgs(cmd *cobra.Command, args []string) error {
	for _, a := range args {
		if a != "qux" {
			return errors.Errorf("arg %s does not equal qux", a)
		}
	}
	return nil
}

func TestBindToCommandArgRequirements(t *testing.T) {

	// because cobra.ExactArgs(1) == cobra.ExactArgs(3), it is needed
	// to run test argument sets that both pass and fail to ensure the correct function was set.
	var usecases = []struct {
		name      string
		runner    Runner
		testCases map[string]argTest
		cmd       *cobra.Command
	}{
		{
			name: "leaf command, no defined args, follow parent",
			runner: Runner{
				Phases: []Phase{phaseBuilder("foo")},
			},
			testCases: map[string]argTest{
				"phase foo": {
					pass: []string{"one", "two", "three"},
					fail: []string{"one", "two"},
					args: cobra.ExactArgs(3),
				},
			},
			cmd: &cobra.Command{
				Use:  "init",
				Args: cobra.ExactArgs(3),
			},
		},
		{
			name: "container cmd expect none, custom arg check for leaf",
			runner: Runner{
				Phases: []Phase{phaseBuilder6("foo", cobra.NoArgs,
					phaseBuilder6("bar", cobra.ExactArgs(1)),
					phaseBuilder6("baz", customArgs),
				)},
			},
			testCases: map[string]argTest{
				"phase foo": {
					pass: []string{},
					fail: []string{"one"},
					args: cobra.NoArgs,
				},
				"phase foo bar": {
					pass: []string{"one"},
					fail: []string{"one", "two"},
					args: cobra.ExactArgs(1),
				},
				"phase foo baz": {
					pass: []string{"qux"},
					fail: []string{"one"},
					args: customArgs,
				},
			},
			cmd: &cobra.Command{
				Use:  "init",
				Args: cobra.NoArgs,
			},
		},
	}

	for _, rt := range usecases {
		t.Run(rt.name, func(t *testing.T) {

			rt.runner.BindToCommand(rt.cmd)

			// Checks that cmd gets a new phase subcommand
			phaseCmd := getCmd(rt.cmd, "phase")
			if phaseCmd == nil {
				t.Error("cmd didn't have phase subcommand\n")
				return
			}

			for c, args := range rt.testCases {

				cCmd := getCmd(rt.cmd, c)
				if cCmd == nil {
					t.Errorf("cmd didn't have %s subcommand\n", c)
					continue
				}

				// Test passing argument set
				err := cCmd.Args(cCmd, args.pass)
				if err != nil {
					t.Errorf("command %s should validate the args: %v\n %v", cCmd.Name(), args.pass, err)
				}

				// Test failing argument set
				err = cCmd.Args(cCmd, args.fail)

				if err == nil {
					t.Errorf("command %s should fail to validate the args: %v\n %v", cCmd.Name(), args.pass, err)
				}
			}

		})
	}
}

func TestBindToCommand(t *testing.T) {

	var dummy string
	localFlags := pflag.NewFlagSet("dummy", pflag.ContinueOnError)
	localFlags.StringVarP(&dummy, "flag4", "d", "d", "d")

	var usecases = []struct {
		name                string
		runner              Runner
		expectedCmdAndFlags map[string][]string
		setAdditionalFlags  func(*pflag.FlagSet)
	}{
		{
			name:   "when there are no phases, cmd should be left untouched",
			runner: Runner{},
		},
		{
			name: "phases should not inherits any parent flags by default",
			runner: Runner{
				Phases: []Phase{phaseBuilder4("foo", nil)},
			},
			expectedCmdAndFlags: map[string][]string{
				"phase foo": {},
			},
		},
		{
			name: "phases should be allowed to select parent flags to inherits",
			runner: Runner{
				Phases: []Phase{phaseBuilder4("foo", []string{"flag1"})},
			},
			expectedCmdAndFlags: map[string][]string{
				"phase foo": {"flag1"}, //not "flag2"
			},
		},
		{
			name: "it should be possible to apply additional flags to all phases",
			runner: Runner{
				Phases: []Phase{
					phaseBuilder4("foo", []string{"flag3"}),
					phaseBuilder4("bar", []string{"flag1", "flag2", "flag3"}),
					phaseBuilder4("baz", []string{"flag1"}), //test if additional flags are filtered too
				},
			},
			setAdditionalFlags: func(flags *pflag.FlagSet) {
				var dummy3 string
				flags.StringVarP(&dummy3, "flag3", "c", "c", "c")
			},
			expectedCmdAndFlags: map[string][]string{
				"phase foo": {"flag3"},
				"phase bar": {"flag1", "flag2", "flag3"},
				"phase baz": {"flag1"},
			},
		},
		{
			name: "it should be possible to apply custom flags to single phases",
			runner: Runner{
				Phases: []Phase{phaseBuilder5("foo", localFlags)},
			},
			expectedCmdAndFlags: map[string][]string{
				"phase foo": {"flag4"},
			},
		},
		{
			name: "all the above applies to nested phases too",
			runner: Runner{
				Phases: []Phase{
					phaseBuilder4("foo", []string{"flag3"},
						phaseBuilder4("bar", []string{"flag1", "flag2", "flag3"}),
						phaseBuilder4("baz", []string{"flag1"}), //test if additional flags are filtered too
						phaseBuilder5("qux", localFlags),
					),
				},
			},
			setAdditionalFlags: func(flags *pflag.FlagSet) {
				var dummy3 string
				flags.StringVarP(&dummy3, "flag3", "c", "c", "c")
			},
			expectedCmdAndFlags: map[string][]string{
				"phase foo":     {"flag3"},
				"phase foo bar": {"flag1", "flag2", "flag3"},
				"phase foo baz": {"flag1"},
				"phase foo qux": {"flag4"},
			},
		},
	}
	for _, rt := range usecases {
		t.Run(rt.name, func(t *testing.T) {

			var dummy1, dummy2 string
			cmd := &cobra.Command{
				Use: "init",
			}

			cmd.Flags().StringVarP(&dummy1, "flag1", "a", "a", "a")
			cmd.Flags().StringVarP(&dummy2, "flag2", "b", "b", "b")

			if rt.setAdditionalFlags != nil {
				rt.runner.SetAdditionalFlags(rt.setAdditionalFlags)
			}

			rt.runner.BindToCommand(cmd)

			// in case of no phases, checks that cmd is untouched
			if len(rt.runner.Phases) == 0 {
				if cmd.Long != "" {
					t.Error("cmd.Long is set while it should be leaved untouched\n")
				}

				if cmd.Flags().Lookup("skip-phases") != nil {
					t.Error("cmd has skip-phases flag while it should not\n")
				}

				if getCmd(cmd, "phase") != nil {
					t.Error("cmd has phase subcommand while it should not\n")
				}

				return
			}

			// Otherwise, if there are phases

			// Checks that cmd get the description set and the skip-phases flags
			if cmd.Long == "" {
				t.Error("cmd.Long not set\n")
			}

			if cmd.Flags().Lookup("skip-phases") == nil {
				t.Error("cmd didn't have skip-phases flag\n")
			}

			// Checks that cmd gets a new phase subcommand (without local flags)
			phaseCmd := getCmd(cmd, "phase")
			if phaseCmd == nil {
				t.Error("cmd didn't have phase subcommand\n")
				return
			}
			if err := cmdHasFlags(phaseCmd); err != nil {
				t.Errorf("command phase didn't have expected flags: %v\n", err)
			}

			// Checks that cmd subcommand gets subcommand for phases (without flags properly sets)
			for c, flags := range rt.expectedCmdAndFlags {

				cCmd := getCmd(cmd, c)
				if cCmd == nil {
					t.Errorf("cmd didn't have %s subcommand\n", c)
					continue
				}

				if err := cmdHasFlags(cCmd, flags...); err != nil {
					t.Errorf("command %s didn't have expected flags: %v\n", c, err)
				}
			}

		})
	}
}

func getCmd(parent *cobra.Command, nestedName string) *cobra.Command {
	names := strings.Split(nestedName, " ")
	for i, n := range names {
		for _, c := range parent.Commands() {
			if c.Name() == n {
				if i == len(names)-1 {
					return c
				}
				parent = c
			}
		}
	}

	return nil
}

func cmdHasFlags(cmd *cobra.Command, expectedFlags ...string) error {
	flags := []string{}
	cmd.Flags().VisitAll(func(f *pflag.Flag) {
		flags = append(flags, f.Name)
	})

	for _, e := range expectedFlags {
		found := false
		for _, f := range flags {
			if f == e {
				found = true
			}
		}
		if !found {
			return errors.Errorf("flag %q does not exists in %s", e, flags)
		}
	}

	if len(flags) != len(expectedFlags) {
		return errors.Errorf("expected flags %s, got %s", expectedFlags, flags)
	}

	return nil
}
