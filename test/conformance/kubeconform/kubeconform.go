/*
Copyright 2020 The Kubernetes Authors.

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
	"flag"
	"fmt"
	"os"
)

// homegrown command structures now but if this grows we may
// want to adopt whatever kubectl uses
type options struct {
	// Flags only used for generating behaviors
	schemaPath string
	resource   string
	area       string

	// Flags only used for linking behaviors
	testdata string
	listAll  bool

	// Flags shared between CLI tools
	behaviorsDir string
}

type actionFunc func(*options) error

func parseFlags() (actionFunc, *options) {
	o := &options{}

	f := flag.NewFlagSet(os.Args[0], flag.ExitOnError)
	f.StringVar(&o.schemaPath, "schema", "", "Path to the OpenAPI schema")
	f.StringVar(&o.resource, "resource", "", "Resource name")
	f.StringVar(&o.area, "area", "", "Area name to use")

	f.StringVar(&o.testdata, "testdata", "test/conformance/testdata/conformance.yaml", "YAML file containing test linkage data")
	f.BoolVar(&o.listAll, "all", false, "List all behaviors, not just those missing tests")

	f.StringVar(&o.behaviorsDir, "dir", "test/conformance/behaviors/", "Path to the behaviors directory")

	f.Usage = func() {
		fmt.Fprintf(os.Stderr,
			"USAGE\n-----\n%s [ options ] { link | gen }\n",
			os.Args[0])
		fmt.Fprintf(os.Stderr, "\nOPTIONS\n-------\n")
		flag.PrintDefaults()
		fmt.Fprintf(os.Stderr, "\nACTIONS\n------------")
		fmt.Fprintf(os.Stderr, `
  'link' lists behaviors associated with tests
  'gen' generates behaviors based on the API schema
`)
	}

	flag.CommandLine = f
	flag.Parse()
	if len(flag.Args()) != 1 {
		flag.CommandLine.Usage()
		os.Exit(2)
	}

	var action actionFunc
	switch flag.Args()[0] {
	case "gen":
		action = gen
		if o.schemaPath == "" {
			action = nil
			fmt.Fprintf(os.Stderr, "-schema is required for 'gen'\n")
		}
		if o.resource == "" {
			action = nil
			fmt.Fprintf(os.Stderr, "-resource is required for 'gen'\n")
		}
		if o.area == "" {
			action = nil
			fmt.Fprintf(os.Stderr, "-area is required for 'gen'\n")
		}
	case "link":
		action = link
		if o.testdata == "" {
			action = nil
			fmt.Fprintf(os.Stderr, "-testdata is required for 'link'\n")
		}
	}

	if o.behaviorsDir == "" {
		action = nil
		fmt.Fprintf(os.Stderr, "-dir is required\n")
	}

	if action == nil {
		flag.CommandLine.Usage()
		os.Exit(2)
	}
	return action, o
}

func main() {
	action, o := parseFlags()
	err := action(o)
	if err != nil {
		fmt.Printf("Error: %s\n", err)
		os.Exit(1)
	}
}
