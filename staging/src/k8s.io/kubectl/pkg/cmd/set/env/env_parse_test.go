/*
Copyright 2017 The Kubernetes Authors.

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

package env

import (
	"fmt"
	"io"
	"reflect"
	"strings"
	"testing"

	v1 "k8s.io/api/core/v1"
)

func ExampleIsEnvironmentArgument_true() {
	test := "returns=true"
	fmt.Println(IsEnvironmentArgument(test))
	// Output: true
}

func ExampleIsEnvironmentArgument_false() {
	test := "returnsfalse"
	fmt.Println(IsEnvironmentArgument(test))
	// Output: false
}

func ExampleSplitEnvironmentFromResources() {
	args := []string{`resource`, "ENV\\=ARG", `ONE\=MORE`, `DASH-`}
	fmt.Println(SplitEnvironmentFromResources(args))
	// Output: [resource] [ENV\=ARG ONE\=MORE DASH-] true
}

func ExampleParseEnv_good_with_stdin() {
	r := strings.NewReader("FROM=READER")
	ss := []string{"ENV=VARIABLE", "ENV.TEST=VARIABLE", "AND=ANOTHER", "REMOVE-", "-"}
	fmt.Println(ParseEnv(ss, r))
	// Output:
	// [{ENV VARIABLE nil} {ENV.TEST VARIABLE nil} {AND ANOTHER nil} {FROM READER nil}] [REMOVE] true <nil>
}

func ExampleParseEnv_good_with_stdin_and_error() {
	r := strings.NewReader("FROM=READER")
	ss := []string{"-", "This not in the key=value format."}
	fmt.Println(ParseEnv(ss, r))
	// Output:
	// [] [] true "This not in the key" is not a valid key name: a valid environment variable name must consist of alphabetic characters, digits, '_', '-', or '.', and must not start with a digit (e.g. 'my.env-name',  or 'MY_ENV.NAME',  or 'MyEnvName1', regex used for validation is '[-._a-zA-Z][-._a-zA-Z0-9]*')
}

func ExampleParseEnv_good_without_stdin() {
	ss := []string{"ENV=VARIABLE", "ENV.TEST=VARIABLE", "AND=ANOTHER", "REMOVE-"}
	fmt.Println(ParseEnv(ss, nil))
	// Output:
	// [{ENV VARIABLE nil} {ENV.TEST VARIABLE nil} {AND ANOTHER nil}] [REMOVE] false <nil>
}

func ExampleParseEnv_bad_first() {
	var r io.Reader
	bad := []string{"This not in the key=value format."}
	fmt.Println(ParseEnv(bad, r))
	// Output:
	// [] [] false "This not in the key" is not a valid key name: a valid environment variable name must consist of alphabetic characters, digits, '_', '-', or '.', and must not start with a digit (e.g. 'my.env-name',  or 'MY_ENV.NAME',  or 'MyEnvName1', regex used for validation is '[-._a-zA-Z][-._a-zA-Z0-9]*')
}

func ExampleParseEnv_bad_second() {
	var r io.Reader
	bad := []string{".=VARIABLE"}
	fmt.Println(ParseEnv(bad, r))
	// Output:
	// [] [] false "." is not a valid key name: must not be '.'
}

func ExampleParseEnv_bad_third() {
	var r io.Reader
	bad := []string{"..=VARIABLE"}
	fmt.Println(ParseEnv(bad, r))
	// Output:
	// [] [] false ".." is not a valid key name: must not be '..'
}

func ExampleParseEnv_bad_fourth() {
	var r io.Reader
	bad := []string{"..ENV=VARIABLE"}
	fmt.Println(ParseEnv(bad, r))
	// Output:
	// [] [] false "..ENV" is not a valid key name: must not start with '..'
}

func TestParseEnvStdin(t *testing.T) {
	tests := []struct {
		name        string
		input       string
		expectedEnv []v1.EnvVar
	}{
		{
			name:  "Case 1: Hash inside a simple value",
			input: "TOKEN=ab#cd\n",
			expectedEnv: []v1.EnvVar{
				{Name: "TOKEN", Value: "ab#cd"},
			},
		},
		{
			name:  "Case 2: Hash inside a database URL",
			input: "DB_URL=postgres://u:p@h/db#x\n",
			expectedEnv: []v1.EnvVar{
				{Name: "DB_URL", Value: "postgres://u:p@h/db#x"},
			},
		},
		{
			name:  "Case 3: Hash inside a password",
			input: "PASSWORD=hello#world\n",
			expectedEnv: []v1.EnvVar{
				{Name: "PASSWORD", Value: "hello#world"},
			},
		},
		{
			name:  "Case 4: Full-line comments",
			input: "# This is a comment\nTOKEN=abc\n",
			expectedEnv: []v1.EnvVar{
				{Name: "TOKEN", Value: "abc"},
			},
		},
		{
			name:  "Case 5: Comment with leading whitespace",
			input: "   # This is a comment\nTOKEN=abc\n",
			expectedEnv: []v1.EnvVar{
				{Name: "TOKEN", Value: "abc"},
			},
		},
		{
			name:  "Case 6: Inline comments preserved as value under Option A semantics",
			input: "MESSAGE=hello # comment\n",
			expectedEnv: []v1.EnvVar{
				{Name: "MESSAGE", Value: "hello # comment"},
			},
		},
		{
			name:  "Case 7: Multiple assignments",
			input: "TOKEN=ab#cd\nPASSWORD=hello#world\nREGION=india\n",
			expectedEnv: []v1.EnvVar{
				{Name: "TOKEN", Value: "ab#cd"},
				{Name: "PASSWORD", Value: "hello#world"},
				{Name: "REGION", Value: "india"},
			},
		},
		{
			name:  "Empty lines and lines with whitespace are ignored",
			input: "\n   \n\t\nTOKEN=abc\n\n",
			expectedEnv: []v1.EnvVar{
				{Name: "TOKEN", Value: "abc"},
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			r := strings.NewReader(tc.input)
			env, remove, usedStdin, err := ParseEnv([]string{"-"}, r)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if !usedStdin {
				t.Errorf("expected usedStdin to be true")
			}
			if len(remove) != 0 {
				t.Errorf("expected no removed variables, got %v", remove)
			}
			if !reflect.DeepEqual(env, tc.expectedEnv) {
				t.Errorf("expected env %+v, got %+v", tc.expectedEnv, env)
			}
		})
	}
}
