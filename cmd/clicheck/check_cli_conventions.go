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

package main

import (
	"fmt"
	"io/ioutil"
	"os"

	cmdsanity "k8s.io/kubectl/pkg/cmd/util/sanity"
	"k8s.io/kubernetes/pkg/kubectl/cmd"
)

func main() {
	var errorCount int

	kubectl := cmd.NewKubectlCommand(os.Stdin, ioutil.Discard, ioutil.Discard)
	errors := cmdsanity.RunCmdChecks(kubectl, cmdsanity.AllCmdChecks, []string{})
	for _, err := range errors {
		errorCount++
		fmt.Fprintf(os.Stderr, "     %d. %s\n", errorCount, err)
	}

	errors = cmdsanity.RunGlobalChecks(cmdsanity.AllGlobalChecks)
	for _, err := range errors {
		errorCount++
		fmt.Fprintf(os.Stderr, "     %d. %s\n", errorCount, err)
	}

	if errorCount > 0 {
		fmt.Fprintf(os.Stdout, "Found %d errors.\n", errorCount)
		os.Exit(1)
	}

	fmt.Fprintln(os.Stdout, "Congrats, CLI looks good!")
}
