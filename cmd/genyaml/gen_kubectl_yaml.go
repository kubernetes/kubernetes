/*
Copyright 2014 The Kubernetes Authors.

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
	"strings"

	"github.com/spf13/cobra"
	"github.com/spf13/pflag"
	"gopkg.in/yaml.v2"
	"k8s.io/kubernetes/cmd/genutils"
	"k8s.io/kubernetes/pkg/kubectl/cmd"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
)

type cmdOption struct {
	Name         string
	Shorthand    string `yaml:",omitempty"`
	DefaultValue string `yaml:"default_value,omitempty"`
	Usage        string `yaml:",omitempty"`
}

type cmdDoc struct {
	Name             string
	Synopsis         string      `yaml:",omitempty"`
	Description      string      `yaml:",omitempty"`
	Options          []cmdOption `yaml:",omitempty"`
	InheritedOptions []cmdOption `yaml:"inherited_options,omitempty"`
	Example          string      `yaml:",omitempty"`
	SeeAlso          []string    `yaml:"see_also,omitempty"`
}

func main() {
	path := "docs/yaml/kubectl"
	if len(os.Args) == 2 {
		path = os.Args[1]
	} else if len(os.Args) > 2 {
		fmt.Fprintf(os.Stderr, "usage: %s [output directory]\n", os.Args[0])
		os.Exit(1)
	}

	outDir, err := genutils.OutDir(path)
	if err != nil {
		fmt.Fprintf(os.Stderr, "failed to get output directory: %v\n", err)
		os.Exit(1)
	}

	// Set environment variables used by kubectl so the output is consistent,
	// regardless of where we run.
	os.Setenv("HOME", "/home/username")
	// TODO os.Stdin should really be something like ioutil.Discard, but a Reader
	kubectl := cmd.NewKubectlCommand(cmdutil.NewFactory(nil), os.Stdin, ioutil.Discard, ioutil.Discard)
	genYaml(kubectl, "", outDir)
	for _, c := range kubectl.Commands() {
		genYaml(c, "kubectl", outDir)
	}
}

// Temporary workaround for yaml lib generating incorrect yaml with long strings
// that do not contain \n.
func forceMultiLine(s string) string {
	if len(s) > 60 && !strings.Contains(s, "\n") {
		s = s + "\n"
	}
	return s
}

func genFlagResult(flags *pflag.FlagSet) []cmdOption {
	result := []cmdOption{}

	flags.VisitAll(func(flag *pflag.Flag) {
		// Todo, when we mark a shorthand is deprecated, but specify an empty message.
		// The flag.ShorthandDeprecated is empty as the shorthand is deprecated.
		// Using len(flag.ShorthandDeprecated) > 0 can't handle this, others are ok.
		if !(len(flag.ShorthandDeprecated) > 0) && len(flag.Shorthand) > 0 {
			opt := cmdOption{
				flag.Name,
				flag.Shorthand,
				flag.DefValue,
				forceMultiLine(flag.Usage),
			}
			result = append(result, opt)
		} else {
			opt := cmdOption{
				Name:         flag.Name,
				DefaultValue: forceMultiLine(flag.DefValue),
				Usage:        forceMultiLine(flag.Usage),
			}
			result = append(result, opt)
		}
	})

	return result
}

func genYaml(command *cobra.Command, parent, docsDir string) {
	doc := cmdDoc{}

	doc.Name = command.Name()
	doc.Synopsis = forceMultiLine(command.Short)
	doc.Description = forceMultiLine(command.Long)

	flags := command.NonInheritedFlags()
	if flags.HasFlags() {
		doc.Options = genFlagResult(flags)
	}
	flags = command.InheritedFlags()
	if flags.HasFlags() {
		doc.InheritedOptions = genFlagResult(flags)
	}

	if len(command.Example) > 0 {
		doc.Example = command.Example
	}

	if len(command.Commands()) > 0 || len(parent) > 0 {
		result := []string{}
		if len(parent) > 0 {
			result = append(result, parent)
		}
		for _, c := range command.Commands() {
			result = append(result, c.Name())
		}
		doc.SeeAlso = result
	}

	final, err := yaml.Marshal(&doc)
	if err != nil {
		fmt.Println(err)
		os.Exit(1)
	}

	var filename string

	if parent == "" {
		filename = docsDir + doc.Name + ".yaml"
	} else {
		filename = docsDir + parent + "_" + doc.Name + ".yaml"
	}

	outFile, err := os.Create(filename)
	if err != nil {
		fmt.Println(err)
		os.Exit(1)
	}
	defer outFile.Close()
	_, err = outFile.Write(final)
	if err != nil {
		fmt.Println(err)
		os.Exit(1)
	}
}
