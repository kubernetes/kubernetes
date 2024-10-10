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

package config

import metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// Preferences stores elements of KubeRC configuration file
type Preferences struct {
	metav1.TypeMeta

	// Overrides is used to change the defaults values of flags of commands.
	// This is especially useful, when user doesn't want to explicitly
	// set flags each time.
	// +optional
	Overrides []CommandOverride

	// Aliases stores the alias definitions. If the alias name collides with
	// a built-in command, built-in command always overrides the alias name.
	// +optional
	Aliases []AliasOverride
}

// AliasOverride stores the alias definitions.
// It is applied in a pre-defined order which is
// kubectl [ALIAS NAME] expands to kubectl [COMMAND] [USER_FLAGS] [USER_EXPLICIT_ARGUMENTS] [USER_KUBERC_ARGUMENTS]
type AliasOverride struct {
	// Name is the name of alias that can only include alphabetical characters
	// If the alias name conflicts with the built-in command,
	// built-in command will be used.
	Name string
	// Command is the single or set of commands to execute, such as "set env" or "create"
	Command string
	// Arguments is allocated for the arguments such as resource names, etc.
	// These arguments are appended after the explicitly defined arguments.
	Args []string
	// Flag is allocated to store the flag definitions of alias
	Flags []CommandOverrideFlag
}

// CommandOverride stores the commands and their associated flag's
// default values.
type CommandOverride struct {
	// Command refers to a command whose flag's default value is changed.
	Command string
	// Flags is a list of flags storing different default values.
	Flags []CommandOverrideFlag
}

// CommandOverrideFlag stores the name and the specified default
// value of the flag.
type CommandOverrideFlag struct {
	// Flags name without dashes as prefix.
	Name string `json:"name"`

	// In a string format of a default value. It will be parsed
	// by kubectl to the compatible value of the flag.
	Default string `json:"default"`
}
