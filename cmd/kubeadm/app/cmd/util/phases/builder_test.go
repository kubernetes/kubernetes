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
	"strings"
	"testing"

	"github.com/spf13/cobra"
)

func TestPhaseCommandUseLine(t *testing.T) {

	var tests = []struct {
		use string
	}{
		{ // #0 Without [phases] > add [phases]
			use: "use",
		},
		{ // #1 With [phases] > don't add [phases] twice
			use: "use [phases]",
		},
	}

	for i, test := range tests {
		p := PhasedCommandBuilder{Use: test.use}

		if strings.Count(p.useLine(), "[phases]") != 1 {
			t.Errorf("Test %d > Invalid useLine string: actual %q, expected %q", i, p.useLine(), "use [phases]")
		}
	}
}

func TestSplitArgs(t *testing.T) {

	// build a phased command with a moked phase map with two phases
	p := PhasedCommandBuilder{phasesMap: map[string]*Phase{"p1": nil, "p2": nil}}

	var tests = []struct {
		args              []string
		expectedPhaseArgs []string
		expectedOtherArgs []string
	}{
		{ // #0 no args > no phaseArgs/no otherArgs
		},
		{ // #1 only phase args > phase args
			args:              []string{"p1", "p2"},
			expectedPhaseArgs: []string{"p1", "p2"},
		},
		{ // #2 only other args > other args
			args:              []string{"o1", "o2"},
			expectedOtherArgs: []string{"o1", "o2"},
		},
		{ // #3 phase args followed by other args > phase args, other args
			args:              []string{"p1", "p2", "o1", "o2"},
			expectedPhaseArgs: []string{"p1", "p2"},
			expectedOtherArgs: []string{"o1", "o2"},
		},
		{ // #4 phase args mixed by other args > phase args, other args
			args:              []string{"o1", "p1", "o2", "p2"},
			expectedPhaseArgs: []string{"p1", "p2"},
			expectedOtherArgs: []string{"o1", "o2"},
		},
	}

	for i, test := range tests {
		actualPhaseArgs, actualOtherArgs := p.splitArgs(test.args)

		if !isSliceOfStringEqual(actualPhaseArgs, test.expectedPhaseArgs) {
			t.Errorf("Test %d > unexpected phaseArgs: actual %v, expected %v", i, actualPhaseArgs, test.expectedPhaseArgs)
		}

		if !isSliceOfStringEqual(actualOtherArgs, test.expectedOtherArgs) {
			t.Errorf("Test %d > unexpected otherArgs: actual %v, expected %v", i, actualOtherArgs, test.expectedOtherArgs)
		}
	}
}

func TestValidateArgs(t *testing.T) {

	// build a phased command with a moked phase map with two phases
	p := PhasedCommandBuilder{phasesMap: map[string]*Phase{"p1": nil, "p2": nil}}

	var tests = []struct {
		args               []string
		phaseArgValidator  cobra.PositionalArgs
		customArgValidator cobra.PositionalArgs
		expectedError      bool
	}{
		{ // #0 no validators > always pass
		},
		{ // #1 phase validator that should pass > pass
			args:              []string{"p1", "p2"},
			phaseArgValidator: cobra.MinimumNArgs(1),
		},
		{ // #2 phase validator that should fail > fails
			args:              []string{"p1", "p2"},
			phaseArgValidator: cobra.MaximumNArgs(1),
			expectedError:     true,
		},
		{ // #3 custom validator that should pass > pass
			args:               []string{"o1", "o2"},
			customArgValidator: cobra.MinimumNArgs(1),
		},
		{ // #4 custom validator that should fail > fails
			args:               []string{"o1", "o2"},
			customArgValidator: cobra.MaximumNArgs(1),
			expectedError:      true,
		},
	}

	for i, test := range tests {
		validator := p.validateArgs(test.phaseArgValidator, test.customArgValidator)

		err := validator(nil, test.args)
		if err != nil {
			if !test.expectedError {
				t.Errorf("Test %d > validateArgs returned unexpected error: %v", i, err)
			}
			continue
		}
		if test.expectedError {
			t.Errorf("Test %d > validateArgs didn't returned error as unexpected", i)
		}
	}
}

type Receiver6 struct{}

func (r Receiver6) RunP1(cmd *cobra.Command, args []string) error { return nil }
func (r Receiver6) RunP2(cmd *cobra.Command, args []string) error { return nil }
func (r Receiver6) RunP3(cmd *cobra.Command, args []string) error { return nil }

func TestSetPhaseTreeAndMap(t *testing.T) {
	var receiver = &Receiver6{}

	var tests = []struct {
		builder           PhasedCommandBuilder
		expectedKeysInMap []string
		expectedError     bool
	}{
		{ // #0 happy case
			builder: PhasedCommandBuilder{Phases: []*Phase{
				{Use: "P1", Run: receiver.RunP1},
				{Use: "P2", Aliases: []string{"X2"}, Run: receiver.RunP2},
			}},
			expectedKeysInMap: []string{"p1", "p2", "x2"},
		},
		{ // #1 phase build fail (in this case does not pass phase validation)
			builder: PhasedCommandBuilder{Phases: []*Phase{
				{Use: "PX"},
			}},
			expectedError: true,
		},
		{ // #2 hidden phase are not included in map (and as a consequence not callable directly)
			builder: PhasedCommandBuilder{Phases: []*Phase{
				{Use: "P1", Run: receiver.RunP1},
				{Use: "P2", Aliases: []string{"X2"}, Run: receiver.RunP2, Hidden: true},
				{Use: "P3", Run: receiver.RunP3},
			}},
			expectedKeysInMap: []string{"p1", "p3"},
		},
		{ // #3 phases with duplicated arg > fails for duplicated args
			builder: PhasedCommandBuilder{Phases: []*Phase{
				{Use: "P1", Run: receiver.RunP1},
				{Use: "P1", Run: receiver.RunP2},
			}},
			expectedError: true,
		},
		{ // #4 phases with duplicated arg / arg aliases > fails for duplicated args
			builder: PhasedCommandBuilder{Phases: []*Phase{
				{Use: "P1"},
				{Use: "P2", Aliases: []string{"P1"}},
			}},
			expectedError: true,
		},
		{ // #5 less than 2 phases > fails
			builder: PhasedCommandBuilder{Phases: []*Phase{
				{Use: "P1"},
			}},
			expectedError: true,
		},
	}

	for i, test := range tests {
		err := test.builder.setPhaseTreeAndMap()
		if err != nil {
			if !test.expectedError {
				t.Errorf("Test %d > setPhaseTreeAndMap returned unexpected error: %v", i, err)
			}
			continue
		}
		if test.expectedError {
			t.Errorf("Test %d > setPhaseTreeAndMap didn't returned error as unexpected", i)
			continue
		}

		// assert phase tree is set
		if test.builder.phasesTree == nil {
			t.Errorf("Test %d > setPhaseTreeAndMap didn't set phaseTree", i)
			continue
		}

		// assert phase map is set
		if test.builder.phasesMap == nil {
			t.Errorf("Test %d > setPhaseTreeAndMap didn't set phaseMap", i)
			continue
		}

		// assert that expected values are in the map
		if len(test.builder.phasesMap) != len(test.expectedKeysInMap) {
			t.Errorf("Test %d > setPhaseTreeAndMap didn't return expected number of keys in phaseMap: actual %d, expected %d", i, len(test.builder.phasesMap), len(test.expectedKeysInMap))
			continue
		}

		for _, k := range test.expectedKeysInMap {
			if _, ok := test.builder.phasesMap[k]; !ok {
				t.Errorf("Test %d > setPhaseTreeAndMap didn't add key %s to phaseMap", i, k)
				continue
			}
		}
	}
}
