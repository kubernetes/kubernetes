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
	"io/ioutil"
	"os"
	"path/filepath"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/kubectl/cmd"
	cmdutil "github.com/GoogleCloudPlatform/kubernetes/pkg/kubectl/cmd/util"
	"github.com/spf13/cobra"
)

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
	kubectl := cmd.NewKubectlCommand(cmdutil.NewFactory(nil), os.Stdin, ioutil.Discard, ioutil.Discard)
	cobra.GenMarkdownTree(kubectl, docsDir)
}
