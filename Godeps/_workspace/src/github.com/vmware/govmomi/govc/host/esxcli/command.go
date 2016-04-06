/*
Copyright (c) 2014 VMware, Inc. All Rights Reserved.

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

package esxcli

import (
	"flag"
	"fmt"
	"strings"

	"github.com/vmware/govmomi/vim25/types"
)

type Command struct {
	name []string
	args []string
}

type CommandInfoItem struct {
	Name        string `xml:"name"`
	DisplayName string `xml:"displayName"`
	Help        string `xml:"help"`
}

type CommandInfoParam struct {
	CommandInfoItem
	Aliases []string `xml:"aliases"`
	Flag    bool     `xml:"flag"`
}

type CommandInfoHint struct {
	Key   string `xml:"key"`
	Value string `xml:"value"`
}

type CommandInfoHints []CommandInfoHint

type CommandInfoMethod struct {
	CommandInfoItem
	Param []CommandInfoParam `xml:"param"`
	Hints CommandInfoHints   `xml:"hints"`
}

type CommandInfo struct {
	CommandInfoItem
	Method []*CommandInfoMethod `xml:"method"`
}

func NewCommand(args []string) *Command {
	c := &Command{}

	for i, arg := range args {
		if strings.HasPrefix(arg, "-") {
			c.args = args[i:]
			break
		} else {
			c.name = append(c.name, arg)
		}
	}

	return c
}

func (c *Command) Namespace() string {
	return strings.Join(c.name[:len(c.name)-1], ".")
}

func (c *Command) Name() string {
	return c.name[len(c.name)-1]
}

func (c *Command) Method() string {
	return "vim.EsxCLI." + strings.Join(c.name, ".")
}

func (c *Command) Moid() string {
	return "ha-cli-handler-" + strings.Join(c.name[:len(c.name)-1], "-")
}

// Parse generates a flag.FlagSet based on the given []CommandInfoParam and
// returns arguments for use with methods.ExecuteSoap
func (c *Command) Parse(params []CommandInfoParam) ([]types.ReflectManagedMethodExecuterSoapArgument, error) {
	flags := flag.NewFlagSet(strings.Join(c.name, " "), flag.ExitOnError)
	vals := make([]string, len(params))

	for i, p := range params {
		v := &vals[i]
		for _, a := range p.Aliases {
			a = strings.TrimPrefix(a[1:], "-")
			flags.StringVar(v, a, "", p.Help)
		}
	}

	err := flags.Parse(c.args)
	if err != nil {
		return nil, err
	}

	args := []types.ReflectManagedMethodExecuterSoapArgument{}

	for i, p := range params {
		if vals[i] == "" {
			continue
		}
		args = append(args, c.Argument(p.Name, vals[i]))
	}

	return args, nil
}

func (c *Command) Argument(name string, val string) types.ReflectManagedMethodExecuterSoapArgument {
	return types.ReflectManagedMethodExecuterSoapArgument{
		Name: name,
		Val:  fmt.Sprintf("<%s>%s</%s>", name, val, name),
	}
}

func (h CommandInfoHints) Formatter() string {
	for _, hint := range h {
		if hint.Key == "formatter" {
			return hint.Value
		}
	}

	return "simple"
}

func (h CommandInfoHints) Fields() []string {
	for _, hint := range h {
		if strings.HasPrefix(hint.Key, "fields:") {
			return strings.Split(hint.Value, ",")
		}
	}

	return nil
}
