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
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"

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

func TestParseEnvWithHashInStdin(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		expected []v1.EnvVar
	}{
		{
			name:  "hash in values is preserved",
			input: "TOKEN=ab#cd\nDB_URL=postgres://u:p@h/db#x\n",
			expected: []v1.EnvVar{
				{Name: "TOKEN", Value: "ab#cd"},
				{Name: "DB_URL", Value: "postgres://u:p@h/db#x"},
			},
		},
		{
			name:     "comment lines are skipped",
			input:    "# leading comment\nTOKEN=value\n \t# indented comment\n",
			expected: []v1.EnvVar{{Name: "TOKEN", Value: "value"}},
		},
		{
			name:     "only a hash comment yields no env vars",
			input:    "# ab=test",
			expected: []v1.EnvVar{},
		},
		{
			name:     "hash after the value is not a comment",
			input:    "TOKEN=ab #cd\n",
			expected: []v1.EnvVar{{Name: "TOKEN", Value: "ab #cd"}},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			env, _, _, err := ParseEnv([]string{"-"}, strings.NewReader(tc.input))
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if diff := cmp.Diff(tc.expected, env); diff != "" {
				t.Errorf("unexpected env vars (-want +got):\n%s", diff)
			}
		})
	}
}
