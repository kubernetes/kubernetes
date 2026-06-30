/*
Copyright 2017 The Kubernetes Authors.

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

package util

import (
	"fmt"
	"sort"
	"strings"

	"k8s.io/klog/v2"

	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	"k8s.io/kubernetes/cmd/kubeadm/app/util/errors"
)

// ArgumentsToCommand takes two Arg slices, one with the base arguments and one
// with optional override arguments. In the return list, base arguments will precede
// override arguments. Depending on MergeMethod, the overrides can append to,
// prepend to, or replace a base argument.
func ArgumentsToCommand(base, overrides []kubeadmapi.Arg) []string {
	// Sort only the base.
	sortArgsSlice(&base)

	// Collect the "replace" overrides.
	overrideArgs := make(map[string]kubeadmapi.Arg, len(overrides))
	tail := make([]string, 0, len(overrides))
	for _, arg := range overrides {
		overrideArgs[arg.Name] = arg
		if arg.MergeMethod == "" {
			tail = append(tail, fmt.Sprintf("--%s=%s", arg.Name, arg.Value))
		}
	}

	command := make([]string, 0, len(base)+len(tail))
	for _, arg := range base {
		ov, ok := overrideArgs[arg.Name]
		switch {
		case !ok:
			// Base arg is unchanged.
		case ov.MergeMethod == "":
			continue
		default:
			arg.Value = kubeadmapi.MergeArgWithBase(arg.Value, ov)
		}
		command = append(command, fmt.Sprintf("--%s=%s", arg.Name, arg.Value))
	}

	return append(command, tail...)
}

// ArgumentsFromCommand parses a CLI command in the form "--foo=bar" to an Arg slice.
// This function's primary purpose is to parse the kubeadm-flags.env file, but can remain unused
// for some releases.
func ArgumentsFromCommand(command []string) []kubeadmapi.Arg {
	args := []kubeadmapi.Arg{}
	for i, arg := range command {
		key, val, err := parseArgument(arg)

		// Ignore if the first argument doesn't satisfy the criteria, it's most often the binary name
		// Warn in all other cases, but don't error out. This can happen only if the user has edited the argument list by hand, so they might know what they are doing
		if err != nil {
			if i != 0 {
				klog.Warningf("[kubeadm] WARNING: The component argument %q could not be parsed correctly. The argument must be of the form %q. Skipping...\n", arg, "--")
			}
			continue
		}

		args = append(args, kubeadmapi.Arg{Name: key, Value: val})
	}

	sortArgsSlice(&args)

	return args
}

// parseArgument parses the argument "--foo=bar" to "foo" and "bar"
func parseArgument(arg string) (string, string, error) {
	if !strings.HasPrefix(arg, "--") {
		return "", "", errors.New("the argument should start with '--'")
	}
	if !strings.Contains(arg, "=") {
		return "", "", errors.New("the argument should have a '=' between the flag and the value")
	}
	// Remove the starting --
	arg = strings.TrimPrefix(arg, "--")
	// Split the string on =. Return only two substrings, since we want only key/value, but the value can include '=' as well
	keyvalSlice := strings.SplitN(arg, "=", 2)

	// Make sure both a key and value is present
	if len(keyvalSlice) != 2 {
		return "", "", errors.New("the argument must have both a key and a value")
	}
	if len(keyvalSlice[0]) == 0 {
		return "", "", errors.New("the argument must have a key")
	}

	return keyvalSlice[0], keyvalSlice[1], nil
}

// sortArgsSlice sorts a slice of Args alpha-numerically.
func sortArgsSlice(argsPtr *[]kubeadmapi.Arg) {
	args := *argsPtr
	sort.Slice(args, func(i, j int) bool {
		if args[i].Name == args[j].Name {
			return args[i].Value < args[j].Value
		}
		return args[i].Name < args[j].Name
	})
}
