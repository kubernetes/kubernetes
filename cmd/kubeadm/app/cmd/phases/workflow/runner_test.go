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
	"errors"
	"fmt"
	"reflect"
	"testing"
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
			err := w.Run()
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
			err := w.Run()
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
				phaseBuilder3("bar", false),
				phaseBuilder3("baz", true),
			),
			phaseBuilder3("qux", false),
		},
	}

	expected := "The \"myCommand\" command executes the following internal workflow:\n" +
		"```\n" +
		"foo   long description for foo ...\n" +
		"  /bar  long description for bar ...\n" +
		"qux   long description for qux ...\n" +
		"```"

	actual := w.Help("myCommand")
	if !reflect.DeepEqual(actual, expected) {
		t.Errorf("\nactual:\n\t%v\nexpected:\n\t%v\n", actual, expected)
	}
}
