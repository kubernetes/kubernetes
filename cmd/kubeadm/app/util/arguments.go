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

	"k8s.io/apimachinery/pkg/util/sets"
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

