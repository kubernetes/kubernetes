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

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	clientcmdapi "k8s.io/client-go/tools/clientcmd/api"
)

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// Preference stores elements of KubeRC configuration file
type Preference struct {
	metav1.TypeMeta

	// Defaults allow changing default option values of commands.
	// This is especially useful, when user doesn't want to explicitly
	// set options each time.
	// +optional
	Defaults []CommandDefaults

	// Aliases allow defining command aliases for existing kubectl commands, with optional default option values.
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
	// +optional
	Aliases []AliasOverride

	// credPluginPolicy specifies the policy governing which, if any, client-go
	// credential plugins may be executed. It MUST be one of { "", "AllowAll", "DenyAll", "Allowlist" }.
	// If the policy is "", then it falls back to "AllowAll" (this is required
	// to maintain backward compatibility). If the policy is DenyAll, no
	// credential plugins may run. If the policy is Allowlist, only those
	// plugins meeting the criteria specified in the `credPluginAllowlist`
	// field may run.
	CredPluginPolicy clientcmdapi.PluginPolicy

	// credPluginAllowlist (the credential plugin allowlist) specifies the
	// conditions under which client-go credential plugins may be executed. If
	// this field is explicitly given the empty list value (`[]`), the user can
	// disallow all plugins. If this field is left unspecified by the user, it
	// will default to `nil`. When the allowlist is `nil`, all binaries will be
	// permited. In order for a credential plugin binary to be allowed, it must
	// match all criteria specified by at least one entry in the allowlist.
	// Curently, the only criteria available is the name of the plugin. Name
	// matching is performed by first resolving the absolute path of both the
	// plugin and the name in the allowlist entry using `exec.LookPath`. It
	// will be called on both, and the resulting strings must be equal.
	//
	// e.g.
	// credPluginAllowlist:
	// - name: cloud-provider-plugin
	// - name: /usr/local/bin/my-plugin
	// In the above example, the user allows the credential plugins
	// `cloud-provider-plugin` (found somewhere in PATH), and the plugin found
	// at the explicit path `/usr/local/bin/my-plugin`.
	CredPluginAllowlist *[]clientcmdapi.AllowlistItem
}

// AliasOverride stores the alias definitions.
type AliasOverride struct {
	// Name is the name of alias that can only include alphabetical characters
	// If the alias name conflicts with the built-in command,
	// built-in command will be used.
	Name string
	// Command is the single or set of commands to execute, such as "set env" or "create"
	Command string
	// PrependArgs stores the arguments such as resource names, etc.
	// These arguments are inserted after the alias name.
	PrependArgs []string
	// AppendArgs stores the arguments such as resource names, etc.
	// These arguments are appended to the USER_ARGS.
	AppendArgs []string
	// Options is allocated to store the option definitions of alias.
	// Options only modify the default value of the option and if
	// user explicitly passes a value, explicit one is used.
	Options []CommandOptionDefault
}

// CommandDefaults stores the commands and their associated option's
// default values.
type CommandDefaults struct {
	// Command refers to a command whose flag's default value is changed.
	Command string
	// Options is a list of options storing different default values.
	Options []CommandOptionDefault
}

// CommandOptionDefault stores the name and the specified default
// value of an option.
type CommandOptionDefault struct {
	// Option name (long form, without dashes).
	Name string

	// In a string format of a default value. It will be parsed
	// by kubectl to the compatible value of the option.
	Default string
}
