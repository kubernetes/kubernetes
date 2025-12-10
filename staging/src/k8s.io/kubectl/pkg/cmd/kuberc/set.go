/*
Copyright The Kubernetes Authors.

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

package kuberc

import (
	"fmt"
	"os"
	"strings"

	"github.com/spf13/cobra"
	"k8s.io/kubectl/pkg/kuberc"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/cli-runtime/pkg/genericiooptions"
	cmdutil "k8s.io/kubectl/pkg/cmd/util"
	"k8s.io/kubectl/pkg/config/v1beta1"
	"k8s.io/kubectl/pkg/util/i18n"
	"k8s.io/kubectl/pkg/util/templates"
	"sigs.k8s.io/yaml"
)

const (
	sectionDefaults = "defaults"
	sectionAliases  = "aliases"
)

var (
	setLong = templates.LongDesc(i18n.T(`
		Set values in the kuberc configuration file.

		Use --section to specify whether to set defaults or aliases.

		For defaults: Sets default flag values for kubectl commands. The --command flag
		should specify only the command (e.g., "get", "create", "set env"), not resources.

		For aliases: Creates command aliases with optional default flag values and arguments.
		Use --prependarg and --appendarg to include resources or other arguments.`))

	setExample = templates.Examples(i18n.T(`
		# Set default output format for 'get' command
		kubectl alpha kuberc set --section defaults --command get --option output=wide

		# Set default output format for a subcommand
		kubectl alpha kuberc set --section defaults --command "set env" --option output=yaml

		# Create an alias 'getn' for 'get' command with prepended 'nodes' resource
		kubectl alpha kuberc set --section aliases --name getn --command get --prependarg nodes --option output=wide

		# Create an alias 'runx' for 'run' command with appended arguments
		kubectl alpha kuberc set --section aliases --name runx --command run --option image=nginx --appendarg "--" --appendarg custom-arg1

		# Overwrite an existing default
		kubectl alpha kuberc set --section defaults --command get --option output=json --overwrite`))
)

// SetOptions contains the options for setting kuberc configuration
type SetOptions struct {
	KubeRCFile  string
	Section     string // "defaults" or "aliases"
	Command     string
	AliasName   string   // for aliases
	Options     []string // flag=value pairs
	PrependArgs []string
	AppendArgs  []string
	Overwrite   bool

	preferences kuberc.PreferencesHandler

	genericiooptions.IOStreams
}

func NewSetOptions(ioStreams genericiooptions.IOStreams) *SetOptions {
	return &SetOptions{
		IOStreams:   ioStreams,
		Options:     []string{},
		PrependArgs: []string{},
		AppendArgs:  []string{},
		preferences: kuberc.NewPreferences(),
	}
}

// NewCmdKubeRCSet returns a Command instance for 'kuberc set' sub command
func NewCmdKubeRCSet(streams genericiooptions.IOStreams) *cobra.Command {
	o := NewSetOptions(streams)

	cmd := &cobra.Command{
		Use:                   "set --section (defaults|aliases) --command COMMAND",
		DisableFlagsInUseLine: true,
		Short:                 i18n.T("Set values in the kuberc configuration"),
		Long:                  setLong,
		Example:               setExample,
		Run: func(cmd *cobra.Command, args []string) {
			cmdutil.CheckErr(o.Complete(cmd))
			cmdutil.CheckErr(o.Validate())
			cmdutil.CheckErr(o.Run())
		},
	}

	o.preferences.AddFlags(cmd.Flags())
	cmd.Flags().StringVar(&o.Section, "section", o.Section, "Section to modify: 'defaults' or 'aliases'")
	cmd.MarkFlagRequired("section") // nolint:errcheck
	cmd.Flags().StringVar(&o.Command, "command", o.Command, "Command to configure (e.g., 'get', 'create', 'set env')")
	cmd.MarkFlagRequired("command") // nolint:errcheck
	cmd.Flags().StringVar(&o.AliasName, "name", o.AliasName, "Alias name (required for --section=aliases)")
	cmd.Flags().StringArrayVar(&o.Options, "option", o.Options, "Flag option in the form flag=value (can be specified multiple times)")
	cmd.Flags().StringArrayVar(&o.PrependArgs, "prependarg", o.PrependArgs, "Argument to prepend to the command (can be specified multiple times, for aliases only)")
	cmd.Flags().StringArrayVar(&o.AppendArgs, "appendarg", o.AppendArgs, "Argument to append to the command (can be specified multiple times, for aliases only)")
	cmd.Flags().BoolVar(&o.Overwrite, "overwrite", o.Overwrite, "Allow overwriting existing entries")

	return cmd
}

// Complete sets default values for SetOptions
func (o *SetOptions) Complete(cmd *cobra.Command) error {
	if cmd.Flags().Changed("kuberc") {
		o.KubeRCFile = cmd.Flag("kuberc").Value.String()
	}

	kubeRCFile, _, err := kuberc.LoadKuberc(o.KubeRCFile)
	if err != nil {
		return err
	}

	o.KubeRCFile = kubeRCFile
	return nil
}

// Validate validates the SetOptions
func (o *SetOptions) Validate() error {
	if o.KubeRCFile == "" {
		return fmt.Errorf("KUBERC is disabled via KUBERC=off environment variable")
	}

	if o.Section != sectionDefaults && o.Section != sectionAliases {
		return fmt.Errorf("--section must be %q or %q, got: %s", sectionDefaults, sectionAliases, o.Section)
	}

	if o.Section == sectionAliases && o.AliasName == "" {
		return fmt.Errorf("--name is required when --section=%s", sectionAliases)
	}

	if o.Section == sectionDefaults && o.AliasName != "" {
		return fmt.Errorf("--name should not be specified when --section=%s", sectionDefaults)
	}

	if o.Section == sectionDefaults && (len(o.PrependArgs) > 0 || len(o.AppendArgs) > 0) {
		return fmt.Errorf("--prependarg and --appendarg are only valid for --section=%s", sectionAliases)
	}

	return nil
}

// Run executes the set command
func (o *SetOptions) Run() error {
	pref, err := o.loadOrCreatePreference()
	if err != nil {
		return err
	}

	optionDefaults, err := o.parseOptions()
	if err != nil {
		return err
	}

	switch o.Section {
	case sectionDefaults:
		err = o.setDefaults(pref, optionDefaults)
	case sectionAliases:
		err = o.setAlias(pref, optionDefaults)
	}
	if err != nil {
		return err
	}

	return kuberc.SavePreference(pref, o.KubeRCFile, o.Out)
}

// loadOrCreatePreference loads existing preference or creates a new one
func (o *SetOptions) loadOrCreatePreference() (*v1beta1.Preference, error) {
	data, err := os.ReadFile(o.KubeRCFile)
	if err != nil && !os.IsNotExist(err) {
		return nil, fmt.Errorf("error reading kuberc file: %w", err)
	}

	if os.IsNotExist(err) || len(data) == 0 {
		return &v1beta1.Preference{
			TypeMeta: metav1.TypeMeta{
				APIVersion: "kubectl.config.k8s.io/v1beta1",
				Kind:       "Preference",
			},
		}, nil
	}

	var pref v1beta1.Preference
	if err := yaml.Unmarshal(data, &pref); err != nil {
		return nil, fmt.Errorf("error parsing kuberc file: %w", err)
	}

	return &pref, nil
}

// parseOptions parses the --option flags into CommandOptionDefault structs
func (o *SetOptions) parseOptions() ([]v1beta1.CommandOptionDefault, error) {
	var options []v1beta1.CommandOptionDefault

	for _, opt := range o.Options {
		parts := strings.SplitN(opt, "=", 2)
		if len(parts) != 2 {
			return nil, fmt.Errorf("invalid option format %q, expected flag=value", opt)
		}

		// Remove leading dashes from flag name if present
		flagName := strings.TrimLeft(parts[0], "-")

		options = append(options, v1beta1.CommandOptionDefault{
			Name:    flagName,
			Default: parts[1],
		})
	}

	return options, nil
}

// setDefaults updates the defaults section of the preference
func (o *SetOptions) setDefaults(pref *v1beta1.Preference, options []v1beta1.CommandOptionDefault) error {
	for i, def := range pref.Defaults {
		if def.Command == o.Command {
			if !o.Overwrite {
				return fmt.Errorf("defaults for command %q already exist, use --overwrite to replace", o.Command)
			}

			if len(options) == 0 {
				// Preserve existing options if --option flag was not specified
				// If specified, completely replace existing options
				options = def.Options
			}

			pref.Defaults[i].Options = options
			return nil
		}
	}

	pref.Defaults = append(pref.Defaults, v1beta1.CommandDefaults{
		Command: o.Command,
		Options: options,
	})

	return nil
}

// setAlias updates the aliases section of the preference
func (o *SetOptions) setAlias(pref *v1beta1.Preference, options []v1beta1.CommandOptionDefault) error {
	// Check if this alias already exists
	for i, alias := range pref.Aliases {
		if alias.Name == o.AliasName {
			if !o.Overwrite {
				return fmt.Errorf("alias %q already exists, use --overwrite to replace", o.AliasName)
			}

			if len(options) == 0 {
				// Preserve existing options if --option flag was not specified
				// If specified, completely replace existing options
				options = alias.Options
			}

			prependArgs := o.PrependArgs
			if len(prependArgs) == 0 {
				// Preserve existing prependArgs if --prependarg flag was not specified
				// If specified, completely replace existing prependArgs
				prependArgs = alias.PrependArgs
			}

			appendArgs := o.AppendArgs
			if len(appendArgs) == 0 {
				// Preserve existing appendArgs if --appendarg flag was not specified
				// If specified, completely replace existing appendArgs
				appendArgs = alias.AppendArgs
			}

			pref.Aliases[i] = v1beta1.AliasOverride{
				Name:        o.AliasName,
				Command:     o.Command,
				PrependArgs: prependArgs,
				AppendArgs:  appendArgs,
				Options:     options,
			}
			return nil
		}
	}

	pref.Aliases = append(pref.Aliases, v1beta1.AliasOverride{
		Name:        o.AliasName,
		Command:     o.Command,
		PrependArgs: o.PrependArgs,
		AppendArgs:  o.AppendArgs,
		Options:     options,
	})

	return nil
}
