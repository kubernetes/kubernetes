/*
Copyright 2023 The Kubernetes Authors.

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

package client

import (
	goflag "flag"
	"github.com/spf13/pflag"
	"k8s.io/code-generator/internal/codegen/execution"
	"k8s.io/code-generator/pkg/codegen/client"
)

// Generator is the interface for generating client code.
type Generator interface {
	Generate(args *client.Args) error
}

// Command is the command for generating client code.
type Command struct {
	Gen Generator
	*goflag.FlagSet
}

func (c Command) Matches(ex *execution.Vars) bool {
	return len(ex.Args) >= 1 && ex.Args[0] == c.Name()
}

func (c Command) Run(ex *execution.Vars) {
	args := &client.Args{}
	fs := pflag.NewFlagSet(ex.Args[0], pflag.ContinueOnError)
	fs.AddGoFlagSet(c.flags()) // make sure we get the klog flags
	defineFlags(fs, args)
	if err := fs.Parse(ex.Args[1:]); err != nil {
		ex.Println("Error parsing arguments:", err)
		ex.Println()
		ex.Println(c.Help())
		ex.Exit(8)
		return
	}
	gen := c.createOrGetGenerator()
	if err := gen.Generate(args); err != nil {
		ex.Println("Error generating clients:", err)
		ex.Exit(9)
		return
	}
}

func (c Command) createOrGetGenerator() Generator {
	if c.Gen == nil {
		return &client.Generator{}
	}
	return c.Gen
}

func (c Command) flags() *goflag.FlagSet {
	if c.FlagSet == nil {
		return goflag.CommandLine
	}
	return c.FlagSet
}
