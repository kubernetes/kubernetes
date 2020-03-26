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
)

type options struct {
	// Flags only used for generating behaviors
	schemaPath string
	resource   string
	area       string

	// Flags only used for linking behaviors
	testdata    string
	listMissing bool

	// Flags shared between CLI tools
	behaviorsDir string
}

func parseFlags() *options {
	o := &options{}

	flag.StringVar(&o.schemaPath, "schema", "", "Path to the OpenAPI schema")
	flag.StringVar(&o.resource, "resource", ".*", "Resource name")
	flag.StringVar(&o.area, "area", "default", "Area name to use")

	flag.StringVar(&o.testdata, "testdata", "../testdata/conformance.yaml", "YAML file containing test linkage data")
	flag.BoolVar(&o.listMissing, "missing", true, "Only list behaviors missing tests")

	flag.StringVar(&o.behaviorsDir, "dir", "../behaviors", "Path to the behaviors directory")

	flag.Parse()
	return o
}

func main() {
	o := parseFlags()
	action := flag.Arg(0)
	if action == "gen" {
		gen(o)
	} else if action == "link" {
		link(o)
	} else {
		fmt.Printf("Unknown argument %s\n", action)
	}
}
