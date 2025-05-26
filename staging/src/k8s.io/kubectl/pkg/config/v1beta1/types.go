/*
Copyright 2025 The Kubernetes Authors.

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

package v1beta1

import metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// Preference stores elements of KubeRC configuration file
type Preference struct {
	metav1.TypeMeta `json:",inline"`

	// defaults allow changing default option values of commands.
	// This is especially useful, when user doesn't want to explicitly
	// set options each time.
	// +listType=atomic
	Defaults []CommandDefaults `json:"defaults"`

	// aliases allow defining command aliases for existing kubectl commands, with optional default option values.
	// If the alias name collides with a built-in command, built-in command always takes precedence.
	// Option overrides defined in the defaults section do NOT apply to aliases for the same command.
	// kubectl [ALIAS NAME] [USER_OPTIONS] [USER_EXPLICIT_ARGS] expands to
	// kubectl [COMMAND] # built-in command alias points to
	//         [KUBERC_PREPEND_ARGS]
	//         [USER_OPTIONS]
	//         [KUBERC_OPTIONS] # rest of the options that are not passed by user in [USER_OPTIONS]
	//         [USER_EXPLICIT_ARGS]
	//         [KUBERC_APPEND_ARGS]
	// e.g.
	// - name: runx
	//   command: run
	//   options:
	//   - name: image
	//     default: nginx
	//   appendArgs:
	//   - --
	//   - custom-arg1
	// For example, if user invokes "kubectl runx test-pod" command,
	// this will be expanded to "kubectl run --image=nginx test-pod -- custom-arg1"
	// - name: getn
	//   command: get
	//   options:
	//   - name: output
	//     default: wide
	//   prependArgs:
	//   - node
	// "kubectl getn control-plane-1" expands to "kubectl get node control-plane-1 --output=wide"
	// "kubectl getn control-plane-1 --output=json" expands to "kubectl get node --output=json control-plane-1"
	// +listType=atomic
	Aliases []AliasOverride `json:"aliases"`
}

// AliasOverride stores the alias definitions.
type AliasOverride struct {
	// name is the name of alias that can only include alphabetical characters
	// If the alias name conflicts with the built-in command,
	// built-in command will be used.
	Name string `json:"name"`
	// command is the single or set of commands to execute, such as "set env" or "create"
	Command string `json:"command"`
	// prependArgs stores the arguments such as resource names, etc.
	// These arguments are inserted after the alias name.
	// +listType=atomic
	PrependArgs []string `json:"prependArgs,omitempty"`
	// appendArgs stores the arguments such as resource names, etc.
	// These arguments are appended to the USER_ARGS.
	// +listType=atomic
	AppendArgs []string `json:"appendArgs,omitempty"`
	// options is allocated to store the option definitions of alias.
	// options only modify the default value of the option and if
	// user explicitly passes a value, explicit one is used.
	// +listType=atomic
	Options []CommandOptionDefault `json:"options,omitempty"`
}

// CommandDefaults stores the commands and their associated option's
// default values.
type CommandDefaults struct {
	// command refers to a command whose option's default value is changed.
	Command string `json:"command"`
	// options is a list of options storing different default values.
	// +listType=atomic
	Options []CommandOptionDefault `json:"options"`
}

// CommandOptionDefault stores the name and the specified default
// value of an option.
type CommandOptionDefault struct {
	// Option name (long form, without dashes).
	Name string `json:"name"`

	// In a string format of a default value. It will be parsed
	// by kubectl to the compatible value of the option.
	Default string `json:"default"`
}
