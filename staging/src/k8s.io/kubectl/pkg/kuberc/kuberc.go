/*
Copyright 2024 The Kubernetes Authors.

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
	"io"
	"os"
	"path/filepath"
	"strings"

	"k8s.io/apimachinery/pkg/runtime"

	"github.com/spf13/cobra"
	"github.com/spf13/pflag"
	"k8s.io/client-go/tools/clientcmd"
	"k8s.io/client-go/util/homedir"
	"k8s.io/kubectl/pkg/cmd/util"
	"k8s.io/kubectl/pkg/config/v1alpha1"
	"k8s.io/kubectl/pkg/scheme"
)

const RecommendedKubeRCFileName = ".kuberc"

var (
	RecommendedConfigDir  = filepath.Join(homedir.HomeDir(), clientcmd.RecommendedHomeDir)
	RecommendedKubeRCFile = filepath.Join(RecommendedConfigDir, RecommendedKubeRCFileName)
)

// PreferencesHandler is responsible for setting default flags
// arguments based on user's kuberc configuration.
type PreferencesHandler interface {
	AddFlags(flags *pflag.FlagSet)
	ApplyOverrides(rootCmd *cobra.Command, args []string, errOut io.Writer) error
	ApplyAliases(rootCmd *cobra.Command, args []string, errOut io.Writer) ([]string, error)
}

// Preferences stores the kuberc file coming either from environment variable
// or file from set in flag or the default kuberc path.
type Preferences struct {
	GetPreferencesFunc func(kuberc string) (*v1alpha1.Preferences, error)

	aliases map[string]struct{}
}

// NewPreferences returns initialized Prefrences object.
func NewPreferences() PreferencesHandler {
	return &Preferences{
		GetPreferencesFunc: DefaultGetPreferences,
		aliases:            make(map[string]struct{}),
	}
}

// AddFlags adds kuberc related flags into the command.
func (p *Preferences) AddFlags(flags *pflag.FlagSet) {
	if util.KubeRC.IsEnabled() {
		flags.String("kuberc", "", "Path to the kuberc file to use for preferences.")
	}
}

// ApplyOverrides injects the default flags defined in kuberc file.
func (p *Preferences) ApplyOverrides(rootCmd *cobra.Command, args []string, errOut io.Writer) error {
	if len(args) <= 1 {
		return nil
	}

	if !util.KubeRC.IsEnabled() {
		return nil
	}

	kubercPath := getExplicitKuberc(args)
	kuberc, err := p.GetPreferencesFunc(kubercPath)
	if err != nil {
		return fmt.Errorf("kuberc error %w", err)
	}

	if kuberc == nil {
		return nil
	}

	args = args[1:]
	cmd, _, err := rootCmd.Find(args)
	if err != nil {
		fmt.Fprintf(errOut, "command not found %v\n", err)
		return nil
	}

	for _, c := range kuberc.Spec.Overrides {
		parsedCmds := strings.Split(c.Command, " ")
		overrideCmd, _, err := rootCmd.Find(parsedCmds)
		if err != nil {
			// this may be referring to the alias command which is not initialized
			// because the actual command is totally different.
			return nil
		}
		if overrideCmd.Name() != cmd.Name() {
			continue
		}

		if _, ok := p.aliases[cmd.Name()]; ok {
			return fmt.Errorf("alias %s can not be overridden", cmd.Name())
		}

		// This function triggers merging the persistent flags in the parent commands.
		_ = cmd.InheritedFlags()

		for _, fl := range c.Flags {
			// explicit flag usage has higher precedence than the kuberc default flag value.
			// We should set the default flag values in kuberc, unless there is any in args.
			if explicitFlagUse(fl.Name, args) {
				continue
			}
			err = cmd.Flags().Set(fl.Name, fl.Default)
			if err != nil {
				return fmt.Errorf("could not apply value %s to flag %s in command %s err: %w", fl.Default, fl.Name, c.Command, err)
			}
		}
	}

	err = cmd.ValidateArgs(args)
	if err != nil {
		return nil
	}

	return nil
}

// ApplyAliases sets all defined aliases in kuberc file first to their corresponding commands.
// Since there may be several alias definitions belonging to the same command, it extracts the
// alias that is currently executed from args. After that it sets the flag definitions in alias as default values
// of the command. Lastly, others parameters (e.g. resources, etc.) that are passed as arguments
// sets to the commands args.
func (p *Preferences) ApplyAliases(rootCmd *cobra.Command, args []string, errOut io.Writer) ([]string, error) {
	if len(args) <= 1 {
		return args, nil
	}

	if !util.KubeRC.IsEnabled() {
		return args, nil
	}

	_, _, err := rootCmd.Find(args[1:])
	if err == nil {
		// Command is found, no need to continue for aliasing
		return args, nil
	}

	kubercPath := getExplicitKuberc(args)
	kuberc, err := p.GetPreferencesFunc(kubercPath)
	if err != nil {
		return args, fmt.Errorf("kuberc error %w", err)
	}

	if kuberc == nil {
		return args, nil
	}

	aliasArgsMap := make(map[string]struct {
		args    []string
		flags   []v1alpha1.PreferencesCommandOverrideFlag
		command *cobra.Command
	})

	for _, alias := range kuberc.Spec.Aliases {
		// do not allow shadowing built-ins
		if _, _, err := rootCmd.Find([]string{alias.Name}); err == nil {
			fmt.Fprintf(errOut, "Setting alias %q to a built-in command is not supported\n", alias.Name)
			continue
		}

		if _, ok := aliasArgsMap[alias.Name]; ok {
			fmt.Fprintf(errOut, "alias %s is already set, skipping...\n", alias.Name)
			continue
		}

		commands := strings.Split(alias.Command, " ")
		existingCmd, flags, err := rootCmd.Find(commands)
		if err != nil {
			fmt.Fprintf(errOut, "command %q not found to set alias %q: %v\n", alias.Command, alias.Name, flags)
			continue
		}

		newCmd := *existingCmd
		newCmd.Use = alias.Name
		aliasCmd := &newCmd

		aliasArgsMap[alias.Name] = struct {
			args    []string
			flags   []v1alpha1.PreferencesCommandOverrideFlag
			command *cobra.Command
		}{
			args:    alias.Arguments,
			flags:   alias.Flags,
			command: aliasCmd,
		}
	}

	// It is verified that all the aliases are valid. So that, now we
	// can define them in to the root command.
	for key, val := range aliasArgsMap {
		p.aliases[key] = struct{}{}
		rootCmd.AddCommand(val.command)
	}

	aliasName := args[1]

	foundAliasCmd, _, err := rootCmd.Find([]string{aliasName})
	if err != nil {
		return args, nil
	}

	aliasArgs, ok := aliasArgsMap[aliasName]
	if !ok {
		return args, nil
	}

	// This function triggers merging the persistent flags in the parent commands.
	_ = foundAliasCmd.InheritedFlags()

	for _, fl := range aliasArgs.flags {
		// explicit flag usage has higher precedence than the kuberc default flag value.
		// We should set the default flag values in kuberc, unless there is any in args.
		if explicitFlagUse(fl.Name, args) {
			continue
		}
		err = foundAliasCmd.Flags().Set(fl.Name, fl.Default)
		if err != nil {
			fmt.Fprintf(errOut, "could not apply value %s to flag %s in alias %s err: %v\n", fl.Default, fl.Name, args[0], err)
			return args, nil
		}
	}

	// all args defined in kuberc should be appended to actual args.
	args = append(args, aliasArgs.args...)
	return args, nil
}

// DefaultGetPreferences returns v1alpha1.KubeRCConfiguration.
// If users sets kuberc file explicitly in --kuberc flag, it has the highest
// priority. If not specified, it looks for in KUBERC environment variable.
// If KUBERC is also not set, it falls back to default .kuberc file at the same location
// where kubeconfig's defaults are residing in.
func DefaultGetPreferences(kuberc string) (*v1alpha1.Preferences, error) {
	if !util.KubeRC.IsEnabled() {
		return nil, nil
	}

	kubeRCFile := RecommendedKubeRCFile
	explicitly := false
	if kuberc != "" {
		kubeRCFile = kuberc
		explicitly = true
	}

	if kubeRCFile == "" && os.Getenv("KUBERC") != "" {
		kubeRCFile = os.Getenv("KUBERC")
		explicitly = true
	}

	kubeRCBytes, err := os.ReadFile(kubeRCFile)
	if err != nil {
		if os.IsNotExist(err) && !explicitly {
			return nil, nil
		}
		return nil, err
	}

	decoded, err := runtime.Decode(scheme.Codecs.UniversalDecoder(v1alpha1.SchemeGroupVersion), kubeRCBytes)
	if err != nil {
		return nil, err
	}
	return decoded.(*v1alpha1.Preferences), nil
}

func getExplicitKuberc(args []string) string {
	var kubercPath string
	for i, arg := range args {
		if arg == "--kuberc" {
			if i+1 < len(args) {
				kubercPath = args[i+1]
				break
			}
		} else if strings.Contains(arg, "--kuberc=") {
			parg := strings.Split(arg, "=")
			if len(parg) > 1 {
				kubercPath = parg[1]
				break
			}
		}
	}

	if kubercPath == "" {
		return ""
	}

	return kubercPath
}

func explicitFlagUse(flagName string, args []string) bool {
	for _, arg := range args {
		if strings.HasPrefix(arg, fmt.Sprintf("--%s", flagName)) || strings.HasPrefix(arg, fmt.Sprintf("--%s=", flagName)) {
			return true
		}
	}
	return false
}
