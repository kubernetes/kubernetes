/*
Copyright 2014 Google Inc. All rights reserved.

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
	"io"
	"os"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/kubectl/cmd"
	"github.com/spf13/cobra"
)

func main() {
	out := os.Stdout
	// Set environment variables used by kubectl so the output is consistent,
	// regardless of where we run.
	os.Setenv("HOME", "/home/username")
	kubectl := cmd.NewFactory(nil).NewKubectlCommand(out)
	fmt.Fprintf(out, "## %s\n\n", kubectl.Name())
	fmt.Fprintf(out, "%s\n\n", kubectl.Short)
	fmt.Fprintln(out, "### Commands\n")
	for _, c := range kubectl.Commands() {
		genMarkdown(c, nil, out)
	}
}

func genMarkdown(command, parent *cobra.Command, out io.Writer) {
	name := command.Name()
	if parent != nil {
		name = fmt.Sprintf("%s %s", parent.Name(), name)
	}
	fmt.Fprintf(out, "#### %s\n", name)
	desc := command.Long
	if len(desc) == 0 {
		desc = command.Short
	}
	fmt.Fprintf(out, "%s\n\n", desc)
	usage := command.UsageString()
	fmt.Fprintf(out, "Usage:\n```\n%s\n```\n\n", usage[9:len(usage)-1])
	for _, c := range command.Commands() {
		genMarkdown(c, command, out)
	}
}
