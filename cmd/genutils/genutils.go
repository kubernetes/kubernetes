/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package genutils

import (
	"fmt"
	"os"
	"path/filepath"

	"github.com/spf13/cobra"
)

func OutDir(path string) (string, error) {
	outDir, err := filepath.Abs(path)
	if err != nil {
		return "", err
	}

	stat, err := os.Stat(outDir)
	if err != nil {
		return "", err
	}

	if !stat.IsDir() {
		return "", fmt.Errorf("output directory %s is not a directory\n", outDir)
	}
	outDir = outDir + "/"
	return outDir, nil
}

func getOutDir(path string, args []string) (string, error) {
	if len(args) == 1 {
		path = args[0]
	} else if len(args) > 1 {
		err := fmt.Errorf("usage: COMMAND [output directory]")
		return "", err
	}

	outDir, err := OutDir(path)
	if err != nil {
		return "", err
	}
	return outDir, nil
}

// AddGeneratedDocsCommands will add 3 new commands to a cobra command:
// genmd, genman, genbash. The commands will be hidden from help output,
// documentation, etc. Each command will generate their respective types of
// documentation and will place those docs in the directory specified by
// their first argument (or they have a fallback/default)
func AddGeneratedDocsCommands(cmd *cobra.Command) {
	genMD := &cobra.Command{
		Use:    "genmd",
		Short:  "genmd",
		Run:    runGenMD,
		Hidden: true,
	}
	cmd.AddCommand(genMD)

	genMan := &cobra.Command{
		Use:    "genman",
		Short:  "genman",
		Run:    runGenMan,
		Hidden: true,
	}
	cmd.AddCommand(genMan)

	genBash := &cobra.Command{
		Use:    "genbash",
		Short:  "genbash",
		Run:    runGenBash,
		Hidden: true,
	}
	cmd.AddCommand(genBash)
}

func runGenMD(cmd *cobra.Command, args []string) {
	root := cmd.Root()
	outDir, err := getOutDir("docs/", args)
	if err != nil {
		fmt.Fprintf(os.Stderr, "%v\n", err)
		os.Exit(1)
	}
	root.GenMarkdownTree(outDir)
}

func runGenMan(cmd *cobra.Command, args []string) {
	header := &cobra.GenManHeader{
		Title:  "KUBERNETES",
		Manual: "Kubernetes User Manual",
	}
	root := cmd.Root()
	outDir, err := getOutDir("docs/man/man1", args)
	if err != nil {
		fmt.Fprintf(os.Stderr, "%v\n", err)
		os.Exit(1)
	}
	root.GenManTree(header, outDir)

}

func runGenBash(cmd *cobra.Command, args []string) {
	root := cmd.Root()
	outDir, err := getOutDir("contrib/completions/bash/", args)
	outFile := filepath.Join(outDir, root.Name())
	if err != nil {
		fmt.Fprintf(os.Stderr, "%v\n", err)
		os.Exit(1)
	}
	root.GenBashCompletionFile(outFile)
}
