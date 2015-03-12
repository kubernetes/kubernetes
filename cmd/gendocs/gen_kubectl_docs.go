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
	"bytes"
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
	"strings"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/kubectl/cmd"
	"github.com/spf13/cobra"
)

func printOptions(out *bytes.Buffer, command *cobra.Command, name string) {
	flags := command.NonInheritedFlags()
	flags.SetOutput(out)
	if flags.HasFlags() {
		fmt.Fprintf(out, "### Options\n\n```\n")
		flags.PrintDefaults()
		fmt.Fprintf(out, "```\n\n")
	}

	parentFlags := command.InheritedFlags()
	parentFlags.SetOutput(out)
	if parentFlags.HasFlags() {
		fmt.Fprintf(out, "### Options inherrited from parent commands\n\n```\n")
		parentFlags.PrintDefaults()
		fmt.Fprintf(out, "```\n\n")
	}
}

func genMarkdown(command *cobra.Command, parent, docsDir string) {
	dparent := strings.Replace(parent, " ", "-", -1)
	name := command.Name()
	dname := name
	if len(parent) > 0 {
		dname = dparent + "-" + name
		name = parent + " " + name
	}

	out := new(bytes.Buffer)
	short := command.Short
	long := command.Long
	if len(long) == 0 {
		long = short
	}

	fmt.Fprintf(out, "## %s\n\n", name)
	fmt.Fprintf(out, "%s\n\n", short)
	fmt.Fprintf(out, "### Synopsis\n\n")
	fmt.Fprintf(out, "\n%s\n\n", long)

	if command.Runnable() {
		fmt.Fprintf(out, "```\n%s\n```\n\n", command.UseLine())
	}

	if len(command.Example) > 0 {
		fmt.Fprintf(out, "### Examples\n\n")
		fmt.Fprintf(out, "```\n%s\n```\n\n", command.Example)
	}

	printOptions(out, command, name)

	if len(command.Commands()) > 0 || len(parent) > 0 {
		fmt.Fprintf(out, "### SEE ALSO\n")
		if len(parent) > 0 {
			link := dparent + ".md"
			fmt.Fprintf(out, "* [%s](%s)\n", dparent, link)
		}
		for _, c := range command.Commands() {
			child := dname + "-" + c.Name()
			link := child + ".md"
			fmt.Fprintf(out, "* [%s](%s)\n", child, link)
			genMarkdown(c, name, docsDir)
		}
		fmt.Fprintf(out, "\n")
	}

	filename := docsDir + dname + ".md"
	outFile, err := os.Create(filename)
	if err != nil {
		fmt.Println(err)
		os.Exit(1)
	}
	defer outFile.Close()
	_, err = outFile.Write(out.Bytes())
	if err != nil {
		fmt.Println(err)
		os.Exit(1)
	}
}

func main() {
	// use os.Args instead of "flags" because "flags" will mess up the man pages!
	docsDir := "docs/man/man1/"
	if len(os.Args) == 2 {
		docsDir = os.Args[1]
	} else if len(os.Args) > 2 {
		fmt.Fprintf(os.Stderr, "usage: %s [output directory]\n", os.Args[0])
		return
	}

	docsDir, err := filepath.Abs(docsDir)
	if err != nil {
		fmt.Fprintf(os.Stderr, err.Error())
		return
	}

	stat, err := os.Stat(docsDir)
	if err != nil {
		fmt.Fprintf(os.Stderr, "output directory %s does not exist\n", docsDir)
		return
	}

	if !stat.IsDir() {
		fmt.Fprintf(os.Stderr, "output directory %s is not a directory\n", docsDir)
		return
	}
	docsDir = docsDir + "/"

	// Set environment variables used by kubectl so the output is consistent,
	// regardless of where we run.
	os.Setenv("HOME", "/home/username")
	//TODO os.Stdin should really be something like ioutil.Discard, but a Reader
	kubectl := cmd.NewFactory(nil).NewKubectlCommand(os.Stdin, ioutil.Discard, ioutil.Discard)
	genMarkdown(kubectl, "", docsDir)
	for _, c := range kubectl.Commands() {
		genMarkdown(c, "kubectl", docsDir)
	}
}
