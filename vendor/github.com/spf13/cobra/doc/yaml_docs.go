// Copyright 2013-2023 The Cobra Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package doc

import (
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/spf13/cobra"
	"github.com/spf13/pflag"
	"gopkg.in/yaml.v3"
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
	Usage            string      `yaml:",omitempty"`
	Options          []cmdOption `yaml:",omitempty"`
	InheritedOptions []cmdOption `yaml:"inherited_options,omitempty"`
	Example          string      `yaml:",omitempty"`
	SeeAlso          []string    `yaml:"see_also,omitempty"`
}

// GenYamlTree creates yaml structured ref files for this command and all descendants
// in the directory given. This function may not work
// correctly if your command names have `-` in them. If you have `cmd` with two
// subcmds, `sub` and `sub-third`, and `sub` has a subcommand called `third`
// it is undefined which help output will be in the file `cmd-sub-third.1`.
func GenYamlTree(cmd *cobra.Command, dir string) error {
	identity := func(s string) string { return s }
	emptyStr := func(s string) string { return "" }
	return GenYamlTreeCustom(cmd, dir, emptyStr, identity)
}

// GenYamlTreeCustom creates yaml structured ref files.
func GenYamlTreeCustom(cmd *cobra.Command, dir string, filePrepender, linkHandler func(string) string) error {
	for _, c := range cmd.Commands() {
		if !c.IsAvailableCommand() || c.IsAdditionalHelpTopicCommand() {
			continue
		}
		if err := GenYamlTreeCustom(c, dir, filePrepender, linkHandler); err != nil {
			return err
		}
	}

	basename := strings.ReplaceAll(cmd.CommandPath(), " ", "_") + ".yaml"
	filename := filepath.Join(dir, basename)
	f, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer f.Close()

	if _, err := io.WriteString(f, filePrepender(filename)); err != nil {
		return err
	}
	if err := GenYamlCustom(cmd, f, linkHandler); err != nil {
		return err
	}
	return nil
}

// GenYaml creates yaml output.
func GenYaml(cmd *cobra.Command, w io.Writer) error {
	return GenYamlCustom(cmd, w, func(s string) string { return s })
}

// GenYamlCustom creates custom yaml output.
func GenYamlCustom(cmd *cobra.Command, w io.Writer, linkHandler func(string) string) error {
	cmd.InitDefaultHelpCmd()
	cmd.InitDefaultHelpFlag()

	yamlDoc := cmdDoc{}
	yamlDoc.Name = cmd.CommandPath()

	yamlDoc.Synopsis = forceMultiLine(cmd.Short)
	yamlDoc.Description = forceMultiLine(cmd.Long)

	if cmd.Runnable() {
		yamlDoc.Usage = cmd.UseLine()
	}

	if len(cmd.Example) > 0 {
		yamlDoc.Example = cmd.Example
	}

	flags := cmd.NonInheritedFlags()
	if flags.HasFlags() {
		yamlDoc.Options = genFlagResult(flags)
	}
	flags = cmd.InheritedFlags()
	if flags.HasFlags() {
		yamlDoc.InheritedOptions = genFlagResult(flags)
	}

	if hasSeeAlso(cmd) {
		result := []string{}
		if cmd.HasParent() {
			parent := cmd.Parent()
			result = append(result, parent.CommandPath()+" - "+parent.Short)
		}
		children := cmd.Commands()
		sort.Sort(byName(children))
		for _, child := range children {
			if !child.IsAvailableCommand() || child.IsAdditionalHelpTopicCommand() {
				continue
			}
			result = append(result, child.CommandPath()+" - "+child.Short)
		}
		yamlDoc.SeeAlso = result
	}

	final, err := yaml.Marshal(&yamlDoc)
	if err != nil {
		fmt.Println(err)
		os.Exit(1)
	}

	if _, err := w.Write(final); err != nil {
		return err
	}
	return nil
}

func genFlagResult(flags *pflag.FlagSet) []cmdOption {
	var result []cmdOption

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
