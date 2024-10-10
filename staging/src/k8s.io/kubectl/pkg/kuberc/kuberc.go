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
	"regexp"
	"strings"

	"k8s.io/kubectl/pkg/config"

	"github.com/spf13/cobra"
	"github.com/spf13/pflag"

	"k8s.io/client-go/tools/clientcmd"
	"k8s.io/client-go/util/homedir"
	"k8s.io/kubectl/pkg/config/v1alpha1"
	"k8s.io/kubectl/pkg/scheme"
)

const RecommendedKubeRCFileName = "kuberc"

var (
	RecommendedConfigDir  = filepath.Join(homedir.HomeDir(), clientcmd.RecommendedHomeDir)
	RecommendedKubeRCFile = filepath.Join(RecommendedConfigDir, RecommendedKubeRCFileName)
)

// PreferencesHandler is responsible for setting default flags
// arguments based on user's kuberc configuration.
type PreferencesHandler interface {
	AddFlags(flags *pflag.FlagSet)
	Apply(rootCmd *cobra.Command, args []string, errOut io.Writer) ([]string, error)
}

// Preferences stores the kuberc file coming either from environment variable
// or file from set in flag or the default kuberc path.
type Preferences struct {
	getPreferencesFunc func(kuberc string) (*config.Preferences, error)

	aliases map[string]struct{}
}

// NewPreferences returns initialized Prefrences object.
func NewPreferences() PreferencesHandler {
	return &Preferences{
		getPreferencesFunc: DefaultGetPreferences,
		aliases:            make(map[string]struct{}),
	}
}

// AddFlags adds kuberc related flags into the command.
func (p *Preferences) AddFlags(flags *pflag.FlagSet) {
	flags.String("kuberc", "", "Path to the kuberc file to use for preferences.")
}

// Apply firstly applies the aliases in the preferences file and secondly overrides
// the default values of flags.
func (p *Preferences) Apply(rootCmd *cobra.Command, args []string, errOut io.Writer) ([]string, error) {
	if len(args) <= 1 {
		return args, nil
	}

	kubercPath := getExplicitKuberc(args)
	kuberc, err := p.getPreferencesFunc(kubercPath)
	if err != nil {
		return args, fmt.Errorf("kuberc error %w", err)
	}

	if kuberc == nil {
		return args, nil
	}

	args, err = p.applyAliases(rootCmd, kuberc, args, errOut)
	if err != nil {
		return args, err
	}
	err = p.applyOverrides(rootCmd, kuberc, args, errOut)
	if err != nil {
		return args, err
	}
	return args, nil
}

// applyOverrides finds the command and sets the defaulted flag values in kuberc.
func (p *Preferences) applyOverrides(rootCmd *cobra.Command, kuberc *config.Preferences, args []string, errOut io.Writer) error {
	args = args[1:]
	cmd, _, err := rootCmd.Find(args)
	if err != nil {
		fmt.Fprintf(errOut, "Warning: command not found %v\n", err)
		return nil
	}

	for _, c := range kuberc.Overrides {
		parsedCmds := strings.Split(c.Command, " ")
		overrideCmd, _, err := rootCmd.Find(parsedCmds)
		if err != nil {
			fmt.Fprintf(errOut, "Warning: command %q not found to set kuberc override\n", c.Command)
			continue
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
			existingFlag := cmd.Flag(fl.Name)
			if existingFlag == nil {
				return fmt.Errorf("invalid flag %s for command %s", fl.Name, c.Command)
			}
			if searchInArgs(existingFlag.Name, existingFlag.Shorthand, args) {
				// Don't modify the value implicitly, if it is passed in args explicitly
				continue
			}
			err = cmd.Flags().Set(fl.Name, fl.Default)
			if err != nil {
				return fmt.Errorf("could not apply override value %s to flag %s in command %s err: %w", fl.Default, fl.Name, c.Command, err)
			}
		}
	}

	return nil
}

// applyAliases firstly appends all defined aliases in kuberc file to the root command.
// Since there may be several alias definitions belonging to the same command, it extracts the
// alias that is currently executed from args. After that it sets the flag definitions in alias as default values
// of the command. Lastly, others parameters (e.g. resources, etc.) that are passed as arguments in kuberc
// is appended into the command args.
func (p *Preferences) applyAliases(rootCmd *cobra.Command, kuberc *config.Preferences, args []string, errOut io.Writer) ([]string, error) {
	_, _, err := rootCmd.Find(args[1:])
	if err == nil {
		// Command is found, no need to continue for aliasing
		return args, nil
	}

	aliasArgsMap := make(map[string]struct {
		args    []string
		flags   []config.CommandOverrideFlag
		command *cobra.Command
	})

	for _, alias := range kuberc.Aliases {
		// do not allow shadowing built-ins
		if _, _, err := rootCmd.Find([]string{alias.Name}); err == nil {
			fmt.Fprintf(errOut, "Warning: Setting alias %q to a built-in command is not supported\n", alias.Name)
			continue
		}

		regex := regexp.MustCompile("^[a-zA-Z]+$")
		if !regex.MatchString(alias.Name) {
			return args, fmt.Errorf("invalid alias name, can only include alphabetical characters")
		}

		if _, ok := aliasArgsMap[alias.Name]; ok {
			fmt.Fprintf(errOut, "alias %s is already set, skipping...\n", alias.Name)
			continue
		}

		commands := strings.Split(alias.Command, " ")
		existingCmd, flags, err := rootCmd.Find(commands)
		if err != nil {
			fmt.Fprintf(errOut, "Warning: command %q not found to set alias %q: %v\n", alias.Command, alias.Name, flags)
			continue
		}

		newCmd := *existingCmd
		newCmd.Use = alias.Name
		aliasCmd := &newCmd

		aliasArgsMap[alias.Name] = struct {
			args    []string
			flags   []config.CommandOverrideFlag
			command *cobra.Command
		}{
			args:    alias.Args,
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

	var aliasName string // first "non-flag" arguments
	for _, arg := range args[1:] {
		if !strings.HasPrefix(arg, "-") {
			aliasName = arg
			break
		}
	}

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
		existingFlag := foundAliasCmd.Flag(fl.Name)
		if existingFlag == nil {
			return args, fmt.Errorf("invalid alias flag %s in alias %s", fl.Name, args[0])
		}
		if searchInArgs(existingFlag.Name, existingFlag.Shorthand, args) {
			// Don't modify the value implicitly, if it is passed in args explicitly
			continue
		}
		err = foundAliasCmd.Flags().Set(fl.Name, fl.Default)
		if err != nil {
			return args, fmt.Errorf("could not apply value %s to flag %s in alias %s err: %w", fl.Default, fl.Name, args[0], err)
		}
	}

	// all args defined in kuberc should be appended to actual args.
	args = append(args, aliasArgs.args...)
	// Cobra (command.go#L1078) appends only root command's args into the actual args and ignores the others.
	// We are appending the additional args defined in kuberc in here and
	// expect that it will be passed along to the actual command.
	rootCmd.SetArgs(args[1:])
	return args, nil
}

// DefaultGetPreferences returns KubeRCConfiguration.
// If users sets kuberc file explicitly in --kuberc flag, it has the highest
// priority. If not specified, it looks for in KUBERC environment variable.
// If KUBERC is also not set, it falls back to default .kuberc file at the same location
// where kubeconfig's defaults are residing in.
func DefaultGetPreferences(kuberc string) (*config.Preferences, error) {
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
	pref := &config.Preferences{}
	_, gvk, err := scheme.Codecs.UniversalDecoder(v1alpha1.SchemeGroupVersion).Decode(kubeRCBytes, nil, pref)
	if err != nil {
		return nil, fmt.Errorf("could not be decoded gvk %s, err: %w", gvk, err)
	}
	return pref, nil
}

// Normally, we should extract this value directly from kuberc flag.
// However, flag values are set during the command execution and
// we are in very early stages to prepare commands prior to execute them.
// Besides, we only need kuberc flag value in this stage.
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

// searchInArgs searches the given key in the args and returns
// true, if it finds. Otherwise, it returns false.
func searchInArgs(flagName string, shorthand string, args []string) bool {
	for _, arg := range args {
		// if flag is set in args in "--flag value" or "--flag=value" format,
		// we should return it as found
		if fmt.Sprintf("--%s", flagName) == arg || strings.HasPrefix(arg, fmt.Sprintf("--%s=", flagName)) {
			return true
		}
		if shorthand != "" {
			// shorthand can only be in "-n value" format
			// it is guaranteed that shorthand is one letter. So that
			// checking just the prefix -oyaml also finds --output.
			if strings.HasPrefix(arg, fmt.Sprintf("-%s", shorthand)) {
				return true
			}
		}
	}
	return false
}
