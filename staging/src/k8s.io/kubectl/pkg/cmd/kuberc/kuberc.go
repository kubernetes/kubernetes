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
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"k8s.io/cli-runtime/pkg/genericclioptions"
	"k8s.io/client-go/tools/clientcmd"
	"sigs.k8s.io/yaml"

	"github.com/ohler55/ojg/jp"
	"github.com/ohler55/ojg/oj"
)

// Example kuberc.yaml file
// ---
//apiVersion: v1alpha1
//kind: Preferences
//command:
//	aliases:
//		getdbprod:
//			command: get pods -l what=database --namespace us-2-production
//	overrides:
//		apply:
//			flags:
//				- name: server-side
//				  default: "true"
//		config:
//			flags:
//				- name: kubeconfig
//				  default: "~/.kube/config"
//			set:
//				flags:
//					- name: set-raw-bytes
//					  default: ""
//			view:
//				flags:
//					- name: raw
//					  default: ""
//		delete:
//			flags:
//				- name: confirm
//				  default: "true"
//		default:
//			flags:
//				- name: exec-auth-allowlist
//				  default: /var/kubectl/exec/...

type Preferences struct {
	Kind       string  `json:"kind,omitempty"`
	APIVersion string  `json:"apiVersion,omitempty"`
	Command    Command `json:"command,omitempty"`
}

type Command struct {
	Aliases   map[string]Alias    `json:"aliases,omitempty"`
	Overrides map[string]Override `json:"omitempty"`
}

type Alias struct {
	Command string           `json:"command,omitempty"`
	Aliases map[string]Alias `json:"omitempty"`
}

type Override struct {
	Flags     []Flag              `json:"flags,omitempty"`
	Overrides map[string]Override `json:"omitempty"`
}

type Flag struct {
	Name    string `json:"name"`
	Default string `json:"default"`
}

// ComposeCmdArgs will rewrite the user provided command in place if there is
// a definition in the kuberc file for either an alias or an override, then
// supply the rewritten args back to the top level kubectl command to be
// run as defined
func ComposeCmdArgs(ioStreams genericclioptions.IOStreams, arguments []string) ([]string, error) {
	var renderedCmd []string
	// Figure out how many subcommands deep we're going before the flags start
	var flagsStart int
	for i, arg := range arguments {
		if string(arg[0]) == "-" {
			flagsStart = i
			break
		}
	}
	if flagsStart == 0 {
		flagsStart = len(arguments)
	}

	subcommands := arguments[:flagsStart]
	userFlags := arguments[flagsStart:]

	// Search for a --kuberc flag that will provide a non-default path to the
	// kuberc file
	kubercPath := filepath.Join(clientcmd.RecommendedConfigDir, "kuberc")
	for i, flag := range userFlags {
		if flag == "--kuberc" {
			kubercPath = userFlags[i+1]

			// remove the kuberc flag and value so it doesn't mess up argument parsing
			userFlags = append(userFlags[:i], userFlags[i+2:]...)
			break
		}
	}

	kuberc, err := LoadKubeRC(kubercPath)
	if err != nil {
		return nil, err
	}

	// Use jsonpath to look up nested fields in the kuberc yaml without a
	// bunch of recursive functions
	kubercYAML, err := oj.Parse(kuberc)
	if err != nil {
		return nil, err
	}
	aliasJsonPath := fmt.Sprintf("command.aliases.%s.command", strings.Join(subcommands, "."))
	x, err := jp.ParseString(aliasJsonPath)
	resultsAlias := x.Get(kubercYAML)

	if resultsAlias != nil {
		renderedCmd = strings.Split(resultsAlias[0].(string), " ")
	} else {
		renderedCmd = append(subcommands, userFlags...)
	}

	for i, arg := range renderedCmd {
		if string(arg[0]) == "-" {
			flagsStart = i
			break
		}
	}

	subcommands = renderedCmd[:flagsStart]
	userFlags = renderedCmd[flagsStart:]

	// Use jsonpath to look up nested fields in the kuberc yaml without a
	// bunch of recursive functions
	overridesJsonPath := fmt.Sprintf("command.overrides.%s.flags", strings.Join(subcommands, "."))
	x, err = jp.ParseString(overridesJsonPath)
	resultsOverride := x.Get(kubercYAML)

	// Get all the flags used in the default command override and populate
	// a new slice with them
	var overrideFlags []string
	for i := range resultsOverride {
		// Get the name of the flag
		overridesJsonPathFlagName := fmt.Sprintf("%s[%d].name", overridesJsonPath, i)
		x, err = jp.ParseString(overridesJsonPathFlagName)
		resultsOverride = x.Get(kubercYAML)

		// Check if this is a short or long flag for proper use of dashes
		if resultsOverride != nil && len(resultsOverride[0].(string)) == 1 {
			overrideFlags = append(overrideFlags, "-"+resultsOverride[0].(string))
		} else if resultsOverride != nil && len(resultsOverride[0].(string)) > 1 {
			overrideFlags = append(overrideFlags, "--"+resultsOverride[0].(string))
		}

		// Add the default value for the flag override
		overridesJsonPathDefault := fmt.Sprintf("%s[%d].default", overridesJsonPath, i)
		x, err = jp.ParseString(overridesJsonPathDefault)
		resultsOverride = x.Get(kubercYAML)
		if resultsOverride != nil && resultsOverride[0].(string) != "" {
			overrideFlags = append(overrideFlags, resultsOverride[0].(string))
		}
	}

	fmt.Fprintf(ioStreams.Out, "override flags are %s\n", strings.Join(overrideFlags, " "))

	// To follow the proper precidencing we must concatenate the slices
	// in the order of subcommands -> kuberc specified override flags
	// -> user supplied flags to ensure that any user supplied flags will
	// be what is used in the event of collision.
	finalFlags := append(overrideFlags, userFlags...)
	renderedCmdStr := append(subcommands, finalFlags...)

	return renderedCmdStr, nil
}

// LoadKubeRC reads kuberc file and stores the values in a Preferences struct
// that is then returned
func LoadKubeRC(path string) ([]byte, error) {
	kubercBytes, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}

	kuberc, err := yaml.YAMLToJSON(kubercBytes)
	if err != nil {
		return nil, err
	}

	return kuberc, nil
}
