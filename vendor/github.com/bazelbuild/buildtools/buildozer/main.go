/*
Copyright 2016 Google Inc. All Rights Reserved.
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
// Entry-point for Buildozer binary.

package main

import (
	"flag"
	"fmt"
	"os"
	"strings"

	"github.com/bazelbuild/buildtools/build"
	"github.com/bazelbuild/buildtools/edit"
	"github.com/bazelbuild/buildtools/tables"
)

var (
	buildVersion     = "redacted"
	buildScmRevision = "redacted"

	version           = flag.Bool("version", false, "Print the version of buildozer")
	stdout            = flag.Bool("stdout", false, "write changed BUILD file to stdout")
	buildifier        = flag.String("buildifier", "", "format output using a specific buildifier binary. If empty, use built-in formatter")
	parallelism       = flag.Int("P", 0, "number of cores to use for concurrent actions")
	numio             = flag.Int("numio", 200, "number of concurrent actions")
	commandsFile      = flag.String("f", "", "file name to read commands from, use '-' for stdin (format:|-separated command line arguments to buildozer, excluding flags)")
	keepGoing         = flag.Bool("k", false, "apply all commands, even if there are failures")
	filterRuleTypes   = stringList("types", "comma-separated list of rule types to change, the default empty list means all rules")
	preferEOLComments = flag.Bool("eol-comments", true, "when adding a new comment, put it on the same line if possible")
	rootDir           = flag.String("root_dir", "", "If present, use this folder rather than $PWD to find the root directory.")
	quiet             = flag.Bool("quiet", false, "suppress informational messages")
	editVariables     = flag.Bool("edit-variables", false, "For attributes that simply assign a variable (e.g. hdrs = LIB_HDRS), edit the build variable instead of appending to the attribute.")
	isPrintingProto   = flag.Bool("output_proto", false, "output serialized devtools.buildozer.Output protos instead of human-readable strings.")
	tablesPath        = flag.String("tables", "", "path to JSON file with custom table definitions which will replace the built-in tables")
	addTablesPath     = flag.String("add_tables", "", "path to JSON file with custom table definitions which will be merged with the built-in tables")

	shortenLabelsFlag  = flag.Bool("shorten_labels", true, "convert added labels to short form, e.g. //foo:bar => :bar")
	deleteWithComments = flag.Bool("delete_with_comments", true, "If a list attribute should be deleted even if there is a comment attached to it")
)

func stringList(name, help string) func() []string {
	f := flag.String(name, "", help)
	return func() []string {
		if *f == "" {
			return nil
		}
		res := strings.Split(*f, ",")
		for i := range res {
			res[i] = strings.TrimSpace(res[i])
		}
		return res
	}
}

func main() {
	flag.Parse()

	if *version {
		fmt.Printf("buildozer version: %s \n", buildVersion)
		fmt.Printf("buildozer scm revision: %s \n", buildScmRevision)
		os.Exit(0)
	}

	if *tablesPath != "" {
		if err := tables.ParseAndUpdateJSONDefinitions(*tablesPath, false); err != nil {
			fmt.Fprintf(os.Stderr, "buildifier: failed to parse %s for -tables: %s\n", *tablesPath, err)
			os.Exit(2)
		}
	}

	if *addTablesPath != "" {
		if err := tables.ParseAndUpdateJSONDefinitions(*addTablesPath, true); err != nil {
			fmt.Fprintf(os.Stderr, "buildifier: failed to parse %s for -add_tables: %s\n", *addTablesPath, err)
			os.Exit(2)
		}
	}

	if !(*shortenLabelsFlag) {
		build.DisableRewrites = []string{"label"}
	}
	edit.ShortenLabelsFlag = *shortenLabelsFlag
	edit.DeleteWithComments = *deleteWithComments
	opts := &edit.Options{
		Stdout:            *stdout,
		Buildifier:        *buildifier,
		Parallelism:       *parallelism,
		NumIO:             *numio,
		CommandsFile:      *commandsFile,
		KeepGoing:         *keepGoing,
		FilterRuleTypes:   filterRuleTypes(),
		PreferEOLComments: *preferEOLComments,
		RootDir:           *rootDir,
		Quiet:             *quiet,
		EditVariables:     *editVariables,
		IsPrintingProto:   *isPrintingProto,
	}
	os.Exit(edit.Buildozer(opts, flag.Args()))
}
