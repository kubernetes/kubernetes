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

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/cli-runtime/pkg/genericiooptions"
	cmdutil "k8s.io/kubectl/pkg/cmd/util"
	"k8s.io/kubectl/pkg/config/v1beta1"
	"k8s.io/kubectl/pkg/kuberc"
	"k8s.io/kubectl/pkg/util/i18n"
	"k8s.io/kubectl/pkg/util/templates"
	"sigs.k8s.io/yaml"
)

const (
	sectionDefaults         = "defaults"
	sectionAliases          = "aliases"
	sectionCredentialPlugin = "credentialplugin"

	allowlistEntryFieldName    = "name"
	allowlistEntryFieldCommand = "command"
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
		kubectl kuberc set --section defaults --command get --option output=wide

		# Set default output format for a subcommand
		kubectl kuberc set --section defaults --command "set env" --option output=yaml

		# Create an alias 'getn' for 'get' command with prepended 'nodes' resource
		kubectl kuberc set --section aliases --name getn --command get --prependarg nodes --option output=wide

		# Create an alias 'runx' for 'run' command with appended arguments
		kubectl kuberc set --section aliases --name runx --command run --option image=nginx --appendarg "--" --appendarg custom-arg1

		# Overwrite an existing default
		kubectl kuberc set --section defaults --command get --option output=json --overwrite

		# Set the credential plugin policy and allowlist
		kubectl kuberc set --section credentialplugin --policy Allowlist --allowlist-entry command=cloud-credential-helper`))
)

// SetOptions contains the options for setting kuberc configuration
type SetOptions struct {
	KubeRCFile       string
	Section          string // "defaults", "aliases", or "credentialplugin"
	Command          string
	AliasName        string   // for aliases
	Options          []string // flag=value pairs
	PrependArgs      []string
	AppendArgs       []string
	Overwrite        bool
	PluginPolicy     string
	AllowlistEntries []string

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
	cmd.Flags().StringVar(&o.Section, "section", o.Section, "Section to modify: 'defaults', 'aliases', or 'credentialplugin'")
	cmd.MarkFlagRequired("section") // nolint:errcheck
	cmd.Flags().StringVar(&o.Command, "command", o.Command, "Command to configure (e.g., 'get', 'create', 'set env')")
	cmd.Flags().StringVar(&o.AliasName, "name", o.AliasName, "Alias name (required for --section=aliases)")
	cmd.Flags().StringVar(&o.PluginPolicy, "policy", o.PluginPolicy, "Plugin policy to use for exec credential plugins, must be one of 'AllowAll', 'DenyAll' or 'Allowlist'")
	cmd.Flags().StringArrayVar(&o.Options, "option", o.Options, "Flag option in the form flag=value (can be specified multiple times)")
	cmd.Flags().StringArrayVar(&o.PrependArgs, "prependarg", o.PrependArgs, "Argument to prepend to the command (can be specified multiple times, for aliases only)")
	cmd.Flags().StringArrayVar(&o.AppendArgs, "appendarg", o.AppendArgs, "Argument to append to the command (can be specified multiple times, for aliases only)")
	cmd.Flags().StringArrayVar(&o.AllowlistEntries, "allowlist-entry", o.AllowlistEntries, "Allowlist entry the form field=value (can be specified multiple times)")
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

	if o.Section != sectionDefaults && o.Section != sectionAliases && o.Section != sectionCredentialPlugin {
		return fmt.Errorf("--section must be %q, %q, or %q, got: %s", sectionDefaults, sectionAliases, sectionCredentialPlugin, o.Section)
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

	if o.Section != sectionCredentialPlugin && o.Command == "" {
		return fmt.Errorf("--command is required when --section=%s or --section=%s", sectionAliases, sectionDefaults)
	}

	if o.Section == sectionCredentialPlugin {
		if o.PluginPolicy == "" {
			return fmt.Errorf("--policy is required when --section=%s", sectionCredentialPlugin)
		}

		switch v1beta1.CredentialPluginPolicy(o.PluginPolicy) {
		case v1beta1.PluginPolicyAllowAll, v1beta1.PluginPolicyDenyAll:
			if len(o.AllowlistEntries) != 0 {
				return fmt.Errorf("--allowlist-entry may only be used when --section=%s and --policy=%s", sectionCredentialPlugin, v1beta1.PluginPolicyAllowlist)
			}
		case v1beta1.PluginPolicyAllowlist:
			if len(o.AllowlistEntries) == 0 {
				return fmt.Errorf("--allowlist-entry is required when --section=%s and --policy=%s", sectionCredentialPlugin, v1beta1.PluginPolicyAllowlist)
			}
		default:
			return fmt.Errorf(" --policy must be  %q, %q, or %q, got: %s", v1beta1.PluginPolicyAllowAll, v1beta1.PluginPolicyDenyAll, v1beta1.PluginPolicyAllowlist, o.PluginPolicy)
		}

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
	case sectionCredentialPlugin:
		err = o.setCredentialPlugin(pref, optionDefaults)
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

// setCredentialPlugin sets the credential plugin policy and (optionally) the allowlist
func (o *SetOptions) setCredentialPlugin(pref *v1beta1.Preference, options []v1beta1.CommandOptionDefault) error {
	policy := v1beta1.CredentialPluginPolicy(o.PluginPolicy)

	switch policy {
	case "":
		return fmt.Errorf("credential plugin policy cannot be empty")
	case v1beta1.PluginPolicyAllowAll, v1beta1.PluginPolicyDenyAll:
		if len(o.AllowlistEntries) != 0 {
			return fmt.Errorf("credential plugin allowlist entries provided when policy was not %q", v1beta1.PluginPolicyAllowlist)
		}
	case v1beta1.PluginPolicyAllowlist:
		entries, err := getAllowlistEntries(o)
		if err != nil {
			return err
		}
		pref.CredentialPluginAllowlist = entries
	default:
		return fmt.Errorf("invalid plugin policy: %q", policy)
	}

	pref.CredentialPluginPolicy = policy
	return nil
}

func getAllowlistEntries(o *SetOptions) ([]v1beta1.AllowlistEntry, error) {
	if len(o.AllowlistEntries) == 0 {
		return nil, fmt.Errorf("--allowlist-entry must be provided when policy is %q", v1beta1.PluginPolicyAllowlist)
	}

	var errs []error
	var entries []v1beta1.AllowlistEntry
	for _, keyValuepairs := range o.AllowlistEntries {
		for keyValuePair := range strings.SplitSeq(keyValuepairs, ",") {
			field, value, hasSeparator := strings.Cut(keyValuePair, "=")
			if !hasSeparator {
				errs = append(errs, fmt.Errorf("improperly formatted allowlist entry: %q", keyValuePair))
				continue
			}

			var entry v1beta1.AllowlistEntry
			switch field {
			case allowlistEntryFieldName:
				errs = append(errs, fmt.Errorf("allowlist entry field %q is deprecated, use %q instead", allowlistEntryFieldName, allowlistEntryFieldCommand))
				continue
			case allowlistEntryFieldCommand:
				entry.Command = value
			default:
				errs = append(errs, fmt.Errorf("unrecognized allowlist entry field: %q", field))
				continue
			}

			if len(strings.TrimSpace(value)) == 0 {
				errs = append(errs, fmt.Errorf("empty value in allowlist entry for field %q", field))
				continue
			}

			entries = append(entries, entry)
		}
	}

	return entries, utilerrors.NewAggregate(errs)
}
