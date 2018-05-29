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

func ExampleIsValidEnvironmentArgument_true() {
	test := "wordcharacters=true"
	fmt.Println(IsValidEnvironmentArgument(test))
	// Output: true
}

func ExampleIsValidEnvironmentArgument_false() {
	test := "not$word^characters=test"
	fmt.Println(IsValidEnvironmentArgument(test))
	// Output: false
}

func ExampleSplitEnvironmentFromResources() {
	args := []string{`resource`, "ENV\\=ARG", `ONE\=MORE`, `DASH-`}
	fmt.Println(SplitEnvironmentFromResources(args))
	// Output: [resource] [ENV\=ARG ONE\=MORE DASH-] true
}

func ExampleParseEnv_good() {
	r := strings.NewReader("FROM=READER")
	ss := []string{"ENV=VARIABLE", "AND=ANOTHER", "REMOVE-", "-"}
	fmt.Println(ParseEnv(ss, r))
	// Output:
	// [{ENV VARIABLE nil} {AND ANOTHER nil} {FROM READER nil}] [REMOVE] <nil>
}

func ExampleParseEnv_bad() {
	var r io.Reader
	bad := []string{"This not in the key=value format."}
	fmt.Println(ParseEnv(bad, r))
	// Output:
	// [] [] environment variables must be of the form key=value and can only contain letters, numbers, and underscores
}
