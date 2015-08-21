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

func getOutDir(path string, args []string) string {
	if len(args) == 1 {
		path = args[0]
	} else if len(args) > 1 {
		fmt.Fprintf(os.Stderr, "usage: COMMAND [output directory]\n")
		os.Exit(1)
	}

	outDir, err := OutDir(path)
	if err != nil {
		fmt.Fprintf(os.Stderr, "failed to get output directory: %v\n", err)
		os.Exit(1)
	}
	return outDir
}

func AddGeneratedDocsCommands(cmd *cobra.Command) {
	genMD := &cobra.Command{
		Use:        "genmd",
		Short:      "genmd",
		Run:        runGenMD,
		Deprecated: "but really writing md docs",
	}
	cmd.AddCommand(genMD)

	genMan := &cobra.Command{
		Use:        "genman",
		Short:      "genman",
		Run:        runGenMan,
		Deprecated: "but really writing man pages",
	}
	cmd.AddCommand(genMan)

	genBash := &cobra.Command{
		Use:        "genbash",
		Short:      "genbash",
		Run:        runGenBash,
		Deprecated: "but really writing bash completions",
	}
	cmd.AddCommand(genBash)
}

func runGenMD(cmd *cobra.Command, args []string) {
	parent := cmd.Parent()
	outDir := getOutDir("docs/", args)
	parent.GenMarkdownTree(outDir)
}

func runGenMan(cmd *cobra.Command, args []string) {
	parent := cmd.Parent()
	outDir := getOutDir("docs/man/man1", args)
	parent.GenManTree("KUBERNETES", outDir)

}

func runGenBash(cmd *cobra.Command, args []string) {
	parent := cmd.Parent()
	outDir := getOutDir("contrib/completions/bash/", args)
	outFile := filepath.Join(outDir, parent.Name())
	parent.GenBashCompletionFile(outFile)
}
