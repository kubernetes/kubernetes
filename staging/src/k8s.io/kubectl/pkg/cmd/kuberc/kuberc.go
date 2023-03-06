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

package kuberc

import (
	"os"
	"reflect"
	"strings"

	"github.com/spf13/cobra"

	"k8s.io/cli-runtime/pkg/genericclioptions"
	"k8s.io/klog/v2"
	"sigs.k8s.io/yaml"
)

// Example kuberc.yaml file
// ---
//apiVersion: v1alpha1
//kind: Preferences
//command:
//  aliases:
//    dbprod:
//      command: get pods
//      flags:
//        - -l what=database
//        - --namespace us-2-production
//    dbdev:
//      command: get pods
//      flags:
//        - -l what=database
//        - --namespace us-2-development
//    raw:
//      command: config view
//      flags:
//        - --raw
//    nginx:
//      command: run
//      arguments:
//        - nginx
//      flags:
//        - --image=nginx
//        - --port=5701
//  overrides:
//    apply:
//      flags:
//        - --server-side=true
//    config set:
//      flags:
//        - --set-raw-bytes
//    config view:
//      flags:
//        - --output=json
//    delete:
//      flags:
//        - --confirm=true
//    default:
//      flags:
//        - --kubeconfig='/Users/marcuspuckett/.kube/config-test'
//        - --namespace='test-ns'
//        - --v=7

const DefaultKubercPath = ".kube/kuberc"

type Preferences struct {
	Kind       string  `json:"kind,omitempty"`
	APIVersion string  `json:"apiVersion,omitempty"`
	Command    Command `json:"command,omitempty"`
}

type Command struct {
	Aliases   map[string]Alias    `json:"aliases,omitempty"`
	Overrides map[string]Override `json:"overrides,omitempty"`
}

type Alias struct {
	Command string   `json:"command,omitempty"`
	Flags   []string `json:"flags,omitempty"`
}

type Override struct {
	Flags []string `json:"flags,omitempty"`
}

// Handler is responsible for injecting aliases for commands and
// setting default flags arguments based on user's kuberc configuration.
type Handler interface {
	InjectAliases(rootCmd *cobra.Command, args []string)
	InjectOverrides(rootCmd *cobra.Command, args []string)
	InjectOverridesRoot(flags *genericclioptions.ConfigFlags, args []string)
}

// DefaultKubercHandler implements AliasHandler
type DefaultKubercHandler struct {
	Command Command
}

// NewDefaultKubercHandler instantiates the DefaultKubercHandler by reading the
// kuberc file.
func NewDefaultKubercHandler(kubercPath string) *DefaultKubercHandler {
	kuberc, err := LoadKubeRC(kubercPath)
	if err != nil {
		return &DefaultKubercHandler{}
	}
	return &DefaultKubercHandler{
		kuberc.Command,
	}
}

func (h *DefaultKubercHandler) InjectAliases(rootCmd *cobra.Command, args []string) {
	// Register all aliases
	for alias, command := range h.Command.Aliases {
		commands := strings.Split(command.Command, " ")
		cmd, _, err := rootCmd.Find(commands)
		if err != nil {
			klog.Warningf("Command %q not found to set alias %q", commands, alias)
			continue
		}

		// do not allow shadowing built-ins
		if _, _, err := rootCmd.Find([]string{alias}); err == nil {
			klog.Warningf("Setting alias %q to a built-in command is not supported", alias)
			continue
		}

		// register alias
		cmd.Aliases = append(cmd.Aliases, alias)

		// inject alias flags if this is the command that is being targetted
		// fullAliasCmdPath is the command path defined in kuberc minus the
		// last command (which is being aliased) and the last arg of the cmdArgs
		// (which is what the alias would be). The cmdArgs will also include
		// kubectl so we will ignore the first entry in that array
		//
		// Example: kuberc defines alias "raw" for "config view" subcommand,
		// the user thus will supply "kubectl config raw" on the command line
		// so we take the command defined in the kuberc file and drop the last
		// subcommand, which is view, and replace it with the alias defined in
		// the kuberc file, giving us "kubectl config raw", then we check to
		// see if that is equal to commands supplied by the user.
		fullAliasCmdPath := append(commands[:len(commands)-1], alias)
		if reflect.DeepEqual(fullAliasCmdPath, args[1:]) {
			klog.V(2).Infof("using alias %q, adding flags...", alias)
			cmd.Flags().Parse(command.Flags)
		}
	}
}

func (h *DefaultKubercHandler) InjectOverrides(rootCmd *cobra.Command, args []string) {
	cmd, _, err := rootCmd.Find(args)
	if err != nil {
		klog.Warningf("could not find command %q", args)
		return
	}
	if flags, found := h.Command.Overrides[strings.Join(args, " ")]; found {
		klog.V(2).Infof("adding default flags %q for command %q", flags.Flags, strings.Join(args, " "))
		if err := cmd.Flags().Parse(flags.Flags); err != nil {
			klog.Warningf("error parsing flags - %q", err)
		}
	}
}

func (h *DefaultKubercHandler) InjectOverridesRoot(flags *genericclioptions.ConfigFlags, args []string) {
	if kubercFlags, found := h.Command.Overrides["default"]; found {
		klog.V(2).Infof("adding default flags %q", kubercFlags.Flags)
		flags.OverwriteDefaultConfigFlags(kubercFlags.Flags, args)
	}
}

// LoadKubeRC reads kuberc file and stores the values in a Preferences struct
// that is then returned
func LoadKubeRC(path string) (Preferences, error) {
	kubercBytes, err := os.ReadFile(path)
	if err != nil {
		return Preferences{}, err
	}

	var preferences Preferences
	// TODO: (mpuckett159) This probably should be UnmarshalStrict but I'm not sure
	if err := yaml.Unmarshal(kubercBytes, &preferences); err != nil {
		klog.Warningf("error unmarshalling the yaml for kuberc")
	}

	return preferences, nil
}
