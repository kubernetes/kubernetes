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

package v1alpha1

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	clientcmdapi "k8s.io/client-go/tools/clientcmd/api"
)

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// Preference stores elements of KubeRC configuration file
type Preference struct {
	metav1.TypeMeta `json:",inline"`

	// overrides allows changing default flag values of commands.
	// This is especially useful, when user doesn't want to explicitly
	// set flags each time.
	// +listType=atomic
	Defaults []CommandDefaults `json:"overrides"`

	// aliases allow defining command aliases for existing kubectl commands, with optional default flag values.
	// If the alias name collides with a built-in command, built-in command always takes precedence.
	// Flag overrides defined in the overrides section do NOT apply to aliases for the same command.
	// kubectl [ALIAS NAME] [USER_FLAGS] [USER_EXPLICIT_ARGS] expands to
	// kubectl [COMMAND] # built-in command alias points to
	//         [KUBERC_PREPEND_ARGS]
	//         [USER_FLAGS]
	//         [KUBERC_FLAGS] # rest of the flags that are not passed by user in [USER_FLAGS]
	//         [USER_EXPLICIT_ARGS]
	//         [KUBERC_APPEND_ARGS]
	// e.g.
	// - name: runx
	//   command: run
	//   flags:
	//   - name: image
	//     default: nginx
	//   appendArgs:
	//   - --
	//   - custom-arg1
	// For example, if user invokes "kubectl runx test-pod" command,
	// this will be expanded to "kubectl run --image=nginx test-pod -- custom-arg1"
	// - name: getn
	//   command: get
	//   flags:
	//   - name: output
	//     default: wide
	//   prependArgs:
	//   - node
	// "kubectl getn control-plane-1" expands to "kubectl get node control-plane-1 --output=wide"
	// "kubectl getn control-plane-1 --output=json" expands to "kubectl get node --output=json control-plane-1"
	// +listType=atomic
	Aliases []AliasOverride `json:"aliases"`

	// credentialPluginPolicy specifies the policy governing which, if any, client-go
	// credential plugins may be executed. It MUST be one of { "", "AllowAll", "DenyAll", "Allowlist" }.
	// If the policy is "", then it falls back to "AllowAll" (this is required
	// to maintain backward compatibility). If the policy is DenyAll, no
	// credential plugins may run. If the policy is Allowlist, only those
	// plugins meeting the criteria specified in the `credPluginAllowlist`
	// field may run.
	CredentialPluginPolicy clientcmdapi.PolicyType `json:"credentialPluginPolicy,omitempty"`

	// Allowlist is a slice of allowlist entries. If any of them is a match,
	// then the executable in question may execute. That is, the result is the
	// logical OR of all entries in the allowlist. This list MUST NOT be
	// supplied if the policy is not "Allowlist".
	//
	// e.g.
	// credPluginAllowlist:
	// - name: cloud-provider-plugin
	// - name: /usr/local/bin/my-plugin
	// In the above example, the user allows the credential plugins
	// `cloud-provider-plugin` (found somewhere in PATH), and the plugin found
	// at the explicit path `/usr/local/bin/my-plugin`.
	CredPluginAllowlist clientcmdapi.Allowlist `json:"credPluginAllowlist,omitempty"`
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
	// flags is allocated to store the flag definitions of alias.
	// flags only modifies the default value of the flag and if
	// user explicitly passes a value, explicit one is used.
	// +listType=atomic
	Options []CommandOptionDefault `json:"flags,omitempty"`
}

// CommandDefaults stores the commands and their associated option's
// default values.
type CommandDefaults struct {
	// command refers to a command whose flag's default value is changed.
	Command string `json:"command"`
	// flags is a list of flags storing different default values.
	// +listType=atomic
	Options []CommandOptionDefault `json:"flags"`
}

// CommandOptionDefault stores the name and the specified default
// value of an option.
type CommandOptionDefault struct {
	// Flag name (long form, without dashes).
	Name string `json:"name"`

	// In a string format of a default value. It will be parsed
	// by kubectl to the compatible value of the flag.
	Default string `json:"default"`
}
