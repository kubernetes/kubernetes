/*
Copyright 2018 The Kubernetes Authors.

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

// Package main implements a command line tool for performing structured
// operations on yaml files.
package main

import (
	"flag"
	"log"

	"sigs.k8s.io/structured-merge-diff/v4/internal/cli"
)

func main() {
	var o cli.Options
	o.AddFlags(flag.CommandLine)
	flag.Parse()

	op, err := o.Resolve()
	if err != nil {
		log.Fatalf("Couldn't understand command line flags: %v", err)
	}

	out, err := o.OpenOutput()
	if err != nil {
		log.Fatalf("Couldn't prepare output: %v", err)
	}

	err = op.Execute(out)
	if err != nil {
		log.Fatalf("Couldn't execute operation: %v", err)
	}
}
