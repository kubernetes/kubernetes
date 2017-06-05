/*
Copyright 2016 The Kubernetes Authors.

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
	"io/ioutil"
	"os"
	"sort"

	"github.com/spf13/cobra"
	"github.com/spf13/pflag"
	"gopkg.in/yaml.v2"

	"k8s.io/kubernetes/pkg/kubectl/cmd"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
)

// gen_slate_yaml creates a yaml representation of the kubectl help commands.  This is to be consumed
// by tools to generate documentation.

var outputFile = flag.String("output", "", "Destination for kubectl yaml representation.")

func main() {
	flag.Parse()

	if len(*outputFile) < 1 {
		fmt.Printf("Must specify --output.\n")
		os.Exit(1)
	}

	// Initialize a kubectl command that we can use to get the help documentation
	kubectl := cmd.NewKubectlCommand(cmdutil.NewFactory(nil), os.Stdin, ioutil.Discard, ioutil.Discard)

	// Create the structural representation
	spec := NewKubectlSpec(kubectl)

	// Write the spec to a file as yaml
	WriteFile(spec)
}

func WriteFile(spec KubectlSpec) {
	// Marshall the yaml
	final, err := yaml.Marshal(&spec)
	if err != nil {
		fmt.Println(err)
		os.Exit(1)
	}

	// Create the file
	outFile, err := os.Create(*outputFile)
	if err != nil {
		fmt.Println(err)
		os.Exit(1)
	}
	defer outFile.Close()

	// Write the file
	_, err = outFile.Write(final)
	if err != nil {
		fmt.Println(err)
		os.Exit(1)
	}
}

func NewKubectlSpec(c *cobra.Command) KubectlSpec {
	return KubectlSpec{
		TopLevelCommandGroups: []TopLevelCommands{NewTopLevelCommands(c.Commands())},
	}
}

func NewTopLevelCommands(cs []*cobra.Command) TopLevelCommands {
	tlc := TopLevelCommands{}
	for _, c := range cs {
		tlc.Commands = append(tlc.Commands, NewTopLevelCommand(c))
	}
	sort.Sort(tlc)
	return tlc
}

func NewTopLevelCommand(c *cobra.Command) TopLevelCommand {
	result := TopLevelCommand{
		MainCommand: NewCommand(c, ""),
	}
	for _, sub := range c.Commands() {
		result.SubCommands = append(result.SubCommands, NewSubCommands(sub, "")...)
	}
	sort.Sort(result.SubCommands)
	return result
}

// Parse the Options
func NewOptions(flags *pflag.FlagSet) Options {
	result := Options{}
	flags.VisitAll(func(flag *pflag.Flag) {
		opt := &Option{
			Name:         flag.Name,
			Shorthand:    flag.Shorthand,
			DefaultValue: flag.DefValue,
			Usage:        flag.Usage,
		}
		result = append(result, opt)
	})
	return result
}

// Parse the Commands
func NewSubCommands(c *cobra.Command, path string) Commands {
	subCommands := Commands{NewCommand(c, path+c.Name())}
	for _, subCommand := range c.Commands() {
		subCommands = append(subCommands, NewSubCommands(subCommand, path+c.Name()+" ")...)
	}
	return subCommands
}

func NewCommand(c *cobra.Command, path string) *Command {
	return &Command{
		Name:             c.Name(),
		Path:             path,
		Description:      c.Long,
		Synopsis:         c.Short,
		Example:          c.Example,
		Options:          NewOptions(c.NonInheritedFlags()),
		InheritedOptions: NewOptions(c.InheritedFlags()),
		Usage:            c.Use,
	}
}

//////////////////////////
// Types
//////////////////////////

type KubectlSpec struct {
	TopLevelCommandGroups []TopLevelCommands `yaml:",omitempty"`
}

type TopLevelCommands struct {
	Commands []TopLevelCommand `yaml:",omitempty"`
}
type TopLevelCommand struct {
	MainCommand *Command `yaml:",omitempty"`
	SubCommands Commands `yaml:",omitempty"`
}

type Options []*Option
type Option struct {
	Name         string `yaml:",omitempty"`
	Shorthand    string `yaml:",omitempty"`
	DefaultValue string `yaml:"default_value,omitempty"`
	Usage        string `yaml:",omitempty"`
}

type Commands []*Command
type Command struct {
	Name             string   `yaml:",omitempty"`
	Path             string   `yaml:",omitempty"`
	Synopsis         string   `yaml:",omitempty"`
	Description      string   `yaml:",omitempty"`
	Options          Options  `yaml:",omitempty"`
	InheritedOptions Options  `yaml:"inherited_options,omitempty"`
	Example          string   `yaml:",omitempty"`
	SeeAlso          []string `yaml:"see_also,omitempty"`
	Usage            string   `yaml:",omitempty"`
}

func (a Options) Len() int      { return len(a) }
func (a Options) Swap(i, j int) { a[i], a[j] = a[j], a[i] }
func (a Options) Less(i, j int) bool {
	return a[i].Name < a[j].Name
}

func (a TopLevelCommands) Len() int      { return len(a.Commands) }
func (a TopLevelCommands) Swap(i, j int) { a.Commands[i], a.Commands[j] = a.Commands[j], a.Commands[i] }
func (a TopLevelCommands) Less(i, j int) bool {
	return a.Commands[i].MainCommand.Path < a.Commands[j].MainCommand.Path
}

func (a Commands) Len() int      { return len(a) }
func (a Commands) Swap(i, j int) { a[i], a[j] = a[j], a[i] }
func (a Commands) Less(i, j int) bool {
	return a[i].Path < a[j].Path
}
