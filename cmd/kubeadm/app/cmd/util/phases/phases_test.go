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

package phases

import (
	"fmt"
	"testing"

	"github.com/spf13/cobra"
)

func TestName(t *testing.T) {

	var tests = []struct {
		use          string
		expectedName string
	}{
		{ // #0 single word --> to lower
			use:          "SingleWord",
			expectedName: "singleword",
		},
		{ // #1 many words --> only first used/to lower
			use:          "Many Words",
			expectedName: "many",
		},
	}

	for i, test := range tests {
		actual := name(test.use)
		if actual != test.expectedName {
			t.Errorf("Test %d > Invalid name: actual %s, expected %s", i, actual, test.expectedName)
		}
	}
}

func TestArg(t *testing.T) {

	var tests = []struct {
		parentArgOrArgAlias string
		nameOrAlias         string
		expectedArg         string
	}{
		{ // #0 only phase name or alias --> to lower
			nameOrAlias: "Phase",
			expectedArg: "phase",
		},
		{ // #1 phase name or alias + parent name or alias --> concatenate/to lower
			parentArgOrArgAlias: "Parent",
			nameOrAlias:         "Phase",
			expectedArg:         "parent/phase",
		},
	}

	for i, test := range tests {
		actual := arg(test.parentArgOrArgAlias, test.nameOrAlias)
		if actual != test.expectedArg {
			t.Errorf("Test %d > Invalid name: actual %s, expected %s", i, actual, test.expectedArg)
		}
	}
}

func TestPhaseUseLine(t *testing.T) {

	var tests = []struct {
		name            string
		alias           []string
		level           int
		expectedUseLine string
	}{
		{ // #0 level <=1: concatenate name and aliases
			name:            "name",
			alias:           []string{"alias1", "alias2"},
			expectedUseLine: "name|alias1|alias2",
		},
		{ // #1 level >1: concatenate name and aliases, prefix by phaseArgSeparator
			name:            "name",
			alias:           []string{"alias1", "alias2"},
			level:           2,
			expectedUseLine: "/name|alias1|alias2",
		},
	}

	for i, test := range tests {
		if useLine(test.name, test.alias, test.level) != test.expectedUseLine {
			t.Errorf("Test %d > Invalid useLine: actual %s, expected %s", i, useLine(test.name, test.alias, test.level), test.expectedUseLine)
		}
	}
}

func TestVisitAll(t *testing.T) {
	var phase = Phase{
		Use: "root",
		Phases: []*Phase{
			{
				Use: "A1",
				Phases: []*Phase{
					{
						Use: "A1.A2",
					},
					{
						Use: "A1.B2",
					},
				},
			},
			{
				Use: "B1",
			},
		},
	}

	var expected = []string{"root", "A1", "A1.A2", "A1.B2", "B1"}
	var actual []string
	phase.visitAll(func(p *Phase) {
		actual = append(actual, p.Use)
	})

	if !isSliceOfStringEqual(actual, expected) {
		t.Errorf("error during visitAll: actual %v, expected %v", actual, expected)
	}
}

// Helper method that executes parts of the build logic (setAttributes) on the hierarchy of phases
func (p *Phase) partialBuild1(parent *Phase) error {
	// sets internal attributes (based on hierarchy)
	if err := p.setAttributes(parent); err != nil {
		return err
	}

	// computes arg and argAlias for child phases
	for _, c := range p.Phases {
		if err := c.partialBuild1(p); err != nil {
			return err
		}
	}

	p.readyToRun = true

	return nil
}

func TestSetAttributes(t *testing.T) {

	var tests = []struct {
		phase              Phase
		expectedLevels     []int
		expectedArgs       []string   // one item per phase
		expectedArgAliases [][]string // one item per phase
	}{
		{ // #0 child key are combined with parent key
			phase: Phase{
				Use: "A1",
				Phases: []*Phase{
					{
						Use: "A2",
					},
				},
			},
			expectedLevels:     []int{0, 1},
			expectedArgs:       []string{"a1", "a1/a2"},
			expectedArgAliases: [][]string{{}, {}},
		},
		{ // #1 child key are combined with parent key and aliases
			phase: Phase{
				Use:     "A1",
				Aliases: []string{"x1"},
				Phases: []*Phase{
					{
						Use: "A2",
					},
				},
			},
			expectedLevels:     []int{0, 1},
			expectedArgs:       []string{"a1", "a1/a2"},
			expectedArgAliases: [][]string{{"x1"}, {"x1/a2"}},
		},
		{ // #2 child key and aliases are combined with parent key
			phase: Phase{
				Use: "A1",
				Phases: []*Phase{
					{
						Use:     "A2",
						Aliases: []string{"y2"},
					},
				},
			},
			expectedLevels:     []int{0, 1},
			expectedArgs:       []string{"a1", "a1/a2"},
			expectedArgAliases: [][]string{{}, {"a1/y2"}},
		},
		{ // #3 child key and aliases are combined with parent key and parent aliases
			phase: Phase{
				Use:     "A1",
				Aliases: []string{"x1"},
				Phases: []*Phase{
					{
						Use:     "A2",
						Aliases: []string{"y2"},
					},
				},
			},
			expectedLevels:     []int{0, 1},
			expectedArgs:       []string{"a1", "a1/a2"},
			expectedArgAliases: [][]string{{"x1"}, {"a1/y2", "x1/a2", "x1/y2"}},
		},
	}

	for i, test := range tests {
		// fakes the
		if err := test.phase.partialBuild1(nil); err != nil {
			t.Errorf("Test %d > partialBuild1 returned unexpected error: %v", i, err)
			continue
		}

		var actualLevels []int
		var actualArgs []string
		var actualArgAliases [][]string
		test.phase.visitAll(func(p *Phase) {
			actualLevels = append(actualLevels, p.level)
			actualArgs = append(actualArgs, p.arg)
			actualArgAliases = append(actualArgAliases, p.argAliases)
		})

		if !isSliceOfIntEqual(actualLevels, test.expectedLevels) {
			t.Errorf("Test %d > unexpected levels: actual %v, expected %v", i, actualLevels, test.expectedLevels)
		}

		if !isSliceOfStringEqual(actualArgs, test.expectedArgs) {
			t.Errorf("Test %d > unexpected args: actual %v, expected %v", i, actualArgs, test.expectedArgs)
		}

		if !isSliceOfStringSliceEqual(actualArgAliases, test.expectedArgAliases) {
			t.Errorf("Test %d > unexpected args Aliases: actual %v, expected %v", i, actualArgAliases, test.expectedArgAliases)
		}
	}

}

type Receiver1 struct{}

func (r Receiver1) RunMethod(cmd *cobra.Command, args []string) error { return nil }

func TestValidate(t *testing.T) {

	var receiver = &Receiver1{}

	var tests = []struct {
		phase          Phase
		expectedResult map[string]bool
		expectedError  bool
	}{
		{ // #0 phase with method / without nested phases > pass
			phase: Phase{
				Use: "A1",
				Run: receiver.RunMethod,
			},
		},
		{ // #1 phase without method / without nested phases  > fail
			phase: Phase{
				Use: "A1",
			},
			expectedError: true,
		},
		{ // #2 phase - with method / with nested phases  > pass
			phase: Phase{
				Use: "A1",
				Run: receiver.RunMethod,
				Phases: []*Phase{
					{
						Use: "A2",
						Run: receiver.RunMethod,
					},
				},
			},
		},
		{ // #3 phase - without method / with nested phases > pass
			phase: Phase{
				Use: "A1",
				Phases: []*Phase{
					{
						Use: "A2",
						Run: receiver.RunMethod,
					},
				},
			},
		},
	}

	for i, test := range tests {
		p := test.phase

		if err := p.partialBuild1(nil); err != nil {
			t.Errorf("Test %d > partialBuild2 returned unexpected error: %v", i, err)
			continue
		}

		err := p.validate(nil)
		if err != nil {
			if !test.expectedError {
				t.Errorf("Test %d > validate returned unexpected error: %v", i, err)
			}
			continue
		}
		if test.expectedError {
			t.Errorf("Test %d > validate didn't returned error as unexpected", i)
		}
	}
}

type Receiver2 struct{}

func (r Receiver2) RunMethod1(cmd *cobra.Command, args []string) error { return nil }
func (r Receiver2) RunMethod2(cmd *cobra.Command, args []string) error { return fmt.Errorf("an error") }
func (r Receiver2) RunMethod3(cmd *cobra.Command, args []string) error { panic("some panic") }

func TestExecuteReturnValue(t *testing.T) {

	var receiver = &Receiver2{}

	var tests = []struct {
		phase         Phase
		expectedError bool
	}{
		{ // #0 method returns nil
			phase: Phase{
				Use: "A1",
				Run: receiver.RunMethod1,
			},
		},
		{ // #1 method returns error
			phase: Phase{
				Use: "A1",
				Run: receiver.RunMethod2,
			},
			expectedError: true,
		},
		{ // #2 method panics (recovers from panic)
			phase: Phase{
				Use: "A1",
				Run: receiver.RunMethod3,
			},
			expectedError: true,
		},
		{ // #3 nested phase error
			phase: Phase{
				Use: "A1",
				Phases: []*Phase{
					{
						Use: "A2",
						Run: receiver.RunMethod2,
					},
				},
			},
			expectedError: true,
		},
	}

	for i, test := range tests {

		if err := test.phase.Build(); err != nil {
			t.Errorf("Test %d > Build returned unexpected error: %v", i, err)
			continue
		}

		err := test.phase.Execute(&cobra.Command{}, []string{})
		if err != nil {
			if !test.expectedError {
				t.Errorf("Test %d > run returned unexpected error: %v", i, err)
			}
			continue
		}
		if test.expectedError {
			t.Errorf("Test %d > run didn't returned error as unexpected", i)
		}
	}
}

type Receiver3 struct {
	a map[string]bool
}

func (r Receiver3) RunA1(cmd *cobra.Command, args []string) error {
	r.a["A1"] = true
	return nil
}
func (r Receiver3) RunA1X(cmd *cobra.Command, args []string) error {
	r.a["A1X"] = true
	return nil
}
func (r Receiver3) RunA1Y(cmd *cobra.Command, args []string) error {
	r.a["A1Y"] = true
	return nil
}

func TestRunNestedPhase(t *testing.T) {

	var receiver = &Receiver3{
		a: map[string]bool{},
	}

	var tests = []struct {
		phase          Phase
		expectedResult map[string]bool
	}{
		{ // #0 phase - with method - without child > method executed
			phase: Phase{
				Use: "A1",
				Run: receiver.RunA1,
			},
			expectedResult: map[string]bool{"A1": true},
		},
		{ // #1 phase - with method - with child > all method executed
			phase: Phase{
				Use: "A1",
				Run: receiver.RunA1,
				Phases: []*Phase{
					{
						Use: "X",
						Run: receiver.RunA1X,
					},
					{
						Use: "Y",
						Run: receiver.RunA1Y,
					},
				},
			},
			expectedResult: map[string]bool{"A1": true, "A1X": true, "A1Y": true},
		},
		{ // #2 phase - without method - with child > child method executed
			phase: Phase{
				Use: "A1",
				Phases: []*Phase{
					{
						Use: "X",
						Run: receiver.RunA1X,
					},
					{
						Use: "Y",
						Run: receiver.RunA1Y,
					},
				},
			},
			expectedResult: map[string]bool{"A1X": true, "A1Y": true},
		},
	}

	for i, test := range tests {

		for i := range receiver.a {
			delete(receiver.a, i)
		}

		if err := test.phase.Build(); err != nil {
			t.Errorf("Test %d > Build returned unexpected error: %v", i, err)
			continue
		}

		if err := test.phase.Execute(&cobra.Command{}, []string{}); err != nil {
			t.Errorf("Test %d > run returned unexpected error: %v", i, err)
			continue
		}

		if !isMapEqual(receiver.a, test.expectedResult) {
			t.Errorf("Test %d > unexpected result: actual %v, expected %v", i, receiver.a, test.expectedResult)
			continue
		}
	}
}

type Receiver4 struct {
	a map[string]bool
}

func (r Receiver4) Run(cmd *cobra.Command, args []string) error {
	r.a["X"] = true
	return nil
}

func (r Receiver4) WorkflowIfTrue(cmd *cobra.Command, args []string) (bool, error) { return true, nil }

func (r Receiver4) WorkflowIfFalse(cmd *cobra.Command, args []string) (bool, error) { return false, nil }

func (r Receiver4) WorkflowIfError(cmd *cobra.Command, args []string) (bool, error) {
	return true, fmt.Errorf("error")
}

func TestWorkflowIf(t *testing.T) {

	var receiver = &Receiver4{
		a: map[string]bool{},
	}

	var tests = []struct {
		phase          Phase
		expectedResult map[string]bool
		expectedError  bool
	}{
		{ // #0 phase - without WorkflowIf - always execute
			phase: Phase{
				Use: "root",
				Phases: []*Phase{
					{
						Use: "X",
						Run: receiver.Run,
					},
				},
			},
			expectedResult: map[string]bool{"X": true},
		},
		{ // #1 phase - with WorkflowIf = true - executed
			phase: Phase{
				Use: "root",
				Phases: []*Phase{
					{
						Use:        "X",
						Run:        receiver.Run,
						WorkflowIf: receiver.WorkflowIfTrue,
					},
				},
			},
			expectedResult: map[string]bool{"X": true},
		},
		{ // #2 phase - with runIf = false - not executed
			phase: Phase{
				Use: "root",
				Phases: []*Phase{
					{
						Use:        "X",
						Run:        receiver.Run,
						WorkflowIf: receiver.WorkflowIfFalse,
					},
				},
			},
		},
		{ // #3 run if error handled
			phase: Phase{
				Use: "root",
				Phases: []*Phase{
					{
						Use:        "X",
						Run:        receiver.Run,
						WorkflowIf: receiver.WorkflowIfError,
					},
				},
			},
			expectedError: true,
		},
	}

	for i, test := range tests {

		for i := range receiver.a {
			delete(receiver.a, i)
		}

		if err := test.phase.Build(); err != nil {
			t.Errorf("Test %d > Build returned unexpected error: %v", i, err)
			continue
		}

		err := test.phase.Execute(&cobra.Command{}, []string{})
		if err != nil {
			if !test.expectedError {
				t.Errorf("Test %d > run returned unexpected error: %v", i, err)
			}
			continue
		}
		if test.expectedError {
			t.Errorf("Test %d > run didn't returned error as unexpected", i)
			continue
		}

		if !isMapEqual(receiver.a, test.expectedResult) {
			t.Errorf("Test %d > unexpected result: actual %v, expected %v", i, receiver.a, test.expectedResult)
			continue
		}
	}
}

func isSliceOfIntEqual(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}

	for i, v := range a {
		if v != b[i] {
			return false
		}
	}

	return true
}

func isSliceOfStringEqual(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}

	for i, v := range a {
		if v != b[i] {
			return false
		}
	}

	return true
}

func isSliceOfStringSliceEqual(a, b [][]string) bool {
	if len(a) != len(b) {
		return false
	}

	for i, v := range a {
		if !isSliceOfStringEqual(v, b[i]) {
			return false
		}
	}

	return true
}

func isMapEqual(a, b map[string]bool) bool {
	if len(a) != len(b) {
		return false
	}

	for i := range a {
		if _, ok := b[i]; !ok {
			return false
		}
	}

	return true
}

func TestWorkflowHasByArgOrAlias(t *testing.T) {

	var PhaseWithWorkflow = Phase{
		Use: "",
		Phases: PhaseWorkflow{
			&Phase{
				Use: "A1",
				Phases: PhaseWorkflow{
					&Phase{
						Use: "A1",
					},
				},
			},
			&Phase{
				Use:     "A2",
				Aliases: []string{"B2"},
			},
		},
	}

	if err := PhaseWithWorkflow.partialBuild1(nil); err != nil {
		t.Fatalf("partialBuild1 returned unexpected error: %v", err)
	}

	var tests = []struct {
		argOrAlias     string
		expectedResult bool
	}{
		{argOrAlias: "a1", expectedResult: true},
		{argOrAlias: "a2", expectedResult: true},
		{argOrAlias: "b2", expectedResult: true},
		{argOrAlias: "a1/a1", expectedResult: true},
		{argOrAlias: "x1", expectedResult: false},
		{argOrAlias: "a1/x1", expectedResult: false},
	}

	for i, test := range tests {
		actual := PhaseWithWorkflow.Phases.HasByArgOrAlias(test.argOrAlias)
		if actual != test.expectedResult {
			t.Errorf("Test %d > unexpected result: actual %v, expected %v", i, actual, test.expectedResult)
		}
	}
}
