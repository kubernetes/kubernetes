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

// Commandline utility for querying arbitrary JSON or YAML files.
package main

import (
	"bufio"
	"fmt"
	"log"
	"os"
	"text/template"

	"github.com/spf13/cobra"
	"github.com/spf13/pflag"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util/yaml"
)

var (
	in    = pflag.StringP("input", "f", "", "File to read data from. Defaults to stdin")
	out   = pflag.StringP("output", "o", "", "File to write output to. Defaults to stdout.")
	query = pflag.StringP("template", "t", "", "File with query template. By Default, template is the first positional argument.")
)

const longHelp = `Parse JSON or YAML input and format it with the given template.
The template is interpreted via the go "text/template" package.
For usage and syntax, see: http://golang.org/pkg/text/template`

func main() {
	cmd := &cobra.Command{
		Use:   "manifest-query [template]",
		Short: "Parse JSON or YAML input and format it with the given template",
		Long:  longHelp,
		Run:   run,
	}

	// This is necessary as github.com/spf13/cobra doesn't support "global"
	// pflags currently.  See https://github.com/spf13/cobra/issues/44.
	util.AddPFlagSetToPFlagSet(pflag.CommandLine, cmd.Flags())

	if err := cmd.Execute(); err != nil {
		os.Exit(1)
	}
}

func run(cmd *cobra.Command, args []string) {
	validate(cmd, args)

	writer := os.Stdout
	if *out != "" {
		var err error
		if writer, err = os.Create(*out); err != nil {
			log.Fatal(err)
		}
		defer writer.Close()
	}
	output := bufio.NewWriter(writer)
	defer output.Flush()

	t, err := queryTemplate(args)
	if err != nil {
		log.Fatal(err)
	}

	// Parse input last, so the command can fail fast when reading from stdin.
	manifest, err := parseInput()
	if err != nil {
		log.Fatal(err)
	}

	if err := t.Execute(output, manifest); err != nil {
		log.Fatal(err)
	}
}

// validate checks for illlegal flags or arguments.
func validate(cmd *cobra.Command, args []string) {
	errMsg := ""
	if *query == "" && len(args) == 0 {
		errMsg = "missing query"
	} else if *query != "" && len(args) != 0 || len(args) > 1 {
		errMsg = "too many queries"
	}
	if errMsg != "" {
		fmt.Printf("%[1]s: %s\nTry '%[1]s --help' for more information.\n", cmd.Name(), errMsg)
		os.Exit(1)
	}
}

// parseInput reads and parses the JSON or YAML input.
// For the type of the returned interface, refer to http://golang.org/pkg/encoding/json/#Unmarshal
func parseInput() (*interface{}, error) {
	input := os.Stdin
	if *in != "" {
		var err error
		if input, err = os.Open(*in); err != nil {
			return nil, err
		}
		defer input.Close()
	}

	decoder := yaml.NewYAMLOrJSONDecoder(input, 4096)
	var manifest interface{}
	err := decoder.Decode(&manifest)
	return &manifest, err
}

func queryTemplate(args []string) (*template.Template, error) {
	if *query != "" {
		return template.ParseFiles(*query)
	}

	// Input should already be validated.
	return template.New("query").Parse(args[0] + "\n")
}
