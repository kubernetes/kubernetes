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

	"github.com/pkg/errors"

	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/klog/v2"
	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
)

// ArgumentsToCommand takes two Arg slices, one with the base arguments and one
// with optional override arguments. In the return list override arguments will precede base
// arguments. If an argument is present in the overrides, it will cause
// all instances of the same argument in the base list to be discarded, leaving
// only the instances of this argument in the overrides to be applied.
func ArgumentsToCommand(base []kubeadmapi.Arg, overrides []kubeadmapi.Arg) []string {
	var command []string
	// Copy the overrides arguments into a new slice.
	args := make([]kubeadmapi.Arg, len(overrides))
	copy(args, overrides)

	// overrideArgs is a set of args which will replace the args defined in the base
	overrideArgs := sets.New[string]()
	for _, arg := range overrides {
		overrideArgs.Insert(arg.Name)
	}

	for _, arg := range base {
		if !overrideArgs.Has(arg.Name) {
			args = append(args, arg)
		}
	}

	sort.Slice(args, func(i, j int) bool {
		if args[i].Name == args[j].Name {
			return args[i].Value < args[j].Value
		}
		return args[i].Name < args[j].Name
	})

	for _, arg := range args {
		command = append(command, fmt.Sprintf("--%s=%s", arg.Name, arg.Value))
	}

	return command
}

// ArgumentsFromCommand parses a CLI command in the form "--foo=bar" to an Arg slice
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

	sort.Slice(args, func(i, j int) bool {
		if args[i].Name == args[j].Name {
			return args[i].Value < args[j].Value
		}
		return args[i].Name < args[j].Name
	})
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
