/*
Copyright 2021 The Kubernetes Authors.

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
	"strings"

	"github.com/spf13/cobra"
	"k8s.io/kubectl/pkg/cmd/apiresources"
	"k8s.io/kubectl/pkg/cmd/get"
	cmdutil "k8s.io/kubectl/pkg/cmd/util"
)

var factory cmdutil.Factory

// SetFactoryForCompletion Store the factory which is needed by the completion functions
// Not all commands have access to the factory, so cannot pass it to the completion functions.
func SetFactoryForCompletion(f cmdutil.Factory) {
	factory = f
}

// ResourceTypeAndNameCompletionFunc Returns a completion function that completes as a first argument
// the resource types that match the toComplete prefix, and all following arguments as resource names that match
// the toComplete prefix.
func ResourceTypeAndNameCompletionFunc(f cmdutil.Factory) func(*cobra.Command, []string, string) ([]string, cobra.ShellCompDirective) {
	return func(cmd *cobra.Command, args []string, toComplete string) ([]string, cobra.ShellCompDirective) {
		var comps []string
		if len(args) == 0 {
			comps = apiresources.CompGetResourceList(f, cmd, toComplete)
		} else {
			comps = get.CompGetResource(f, cmd, args[0], toComplete)
			if len(args) > 1 {
				comps = cmdutil.Difference(comps, args[1:])
			}
		}
		return comps, cobra.ShellCompDirectiveNoFileComp
	}
}

// SpecifiedResourceTypeAndNameCompletionFunc Returns a completion function that completes as a first
// argument the resource types that match the toComplete prefix and are limited to the allowedTypes,
// and all following arguments as resource names that match the toComplete prefix.
func SpecifiedResourceTypeAndNameCompletionFunc(f cmdutil.Factory, allowedTypes []string) func(*cobra.Command, []string, string) ([]string, cobra.ShellCompDirective) {
	return doSpecifiedResourceTypeAndNameComp(f, allowedTypes, true)
}

// SpecifiedResourceTypeAndNameNoRepeatCompletionFunc Returns a completion function that completes as a first
// argument the resource types that match the toComplete prefix and are limited to the allowedTypes, and as
// a second argument a resource name that match the toComplete prefix.
func SpecifiedResourceTypeAndNameNoRepeatCompletionFunc(f cmdutil.Factory, allowedTypes []string) func(*cobra.Command, []string, string) ([]string, cobra.ShellCompDirective) {
	return doSpecifiedResourceTypeAndNameComp(f, allowedTypes, false)
}

func doSpecifiedResourceTypeAndNameComp(f cmdutil.Factory, allowedTypes []string, repeat bool) func(*cobra.Command, []string, string) ([]string, cobra.ShellCompDirective) {
	return func(cmd *cobra.Command, args []string, toComplete string) ([]string, cobra.ShellCompDirective) {
		var comps []string
		if len(args) == 0 {
			for _, comp := range allowedTypes {
				if strings.HasPrefix(comp, toComplete) {
					comps = append(comps, comp)
				}
			}
		} else {
			if repeat || len(args) == 1 {
				comps = get.CompGetResource(f, cmd, args[0], toComplete)
				if repeat && len(args) > 1 {
					comps = cmdutil.Difference(comps, args[1:])
				}
			}
		}
		return comps, cobra.ShellCompDirectiveNoFileComp
	}
}

// ResourceNameCompletionFunc Returns a completion function that completes as a first argument
// the resource names specified by the resourceType parameter, and which match the toComplete prefix.
func ResourceNameCompletionFunc(f cmdutil.Factory, resourceType string) func(*cobra.Command, []string, string) ([]string, cobra.ShellCompDirective) {
	return func(cmd *cobra.Command, args []string, toComplete string) ([]string, cobra.ShellCompDirective) {
		var comps []string
		if len(args) == 0 {
			comps = get.CompGetResource(f, cmd, resourceType, toComplete)
		}
		return comps, cobra.ShellCompDirectiveNoFileComp
	}
}

// PodResourceNameAndContainerCompletionFunc Returns a completion function that completes as a first
// argument pod names that match the toComplete prefix, and as a second argument the containers
// within the specified pod.
func PodResourceNameAndContainerCompletionFunc(f cmdutil.Factory) func(*cobra.Command, []string, string) ([]string, cobra.ShellCompDirective) {
	return func(cmd *cobra.Command, args []string, toComplete string) ([]string, cobra.ShellCompDirective) {
		var comps []string
		if len(args) == 0 {
			comps = get.CompGetResource(f, cmd, "pod", toComplete)
		} else if len(args) == 1 {
			comps = get.CompGetContainers(f, cmd, args[0], toComplete)
		}
		return comps, cobra.ShellCompDirectiveNoFileComp
	}
}

// ContextCompletionFunc is a completion function that completes as a first argument the
// context names that match the toComplete prefix
func ContextCompletionFunc(cmd *cobra.Command, args []string, toComplete string) ([]string, cobra.ShellCompDirective) {
	if len(args) == 0 {
		return ListContextsInConfig(toComplete), cobra.ShellCompDirectiveNoFileComp
	}
	return nil, cobra.ShellCompDirectiveNoFileComp
}

// ClusterCompletionFunc is a completion function that completes as a first argument the
// cluster names that match the toComplete prefix
func ClusterCompletionFunc(cmd *cobra.Command, args []string, toComplete string) ([]string, cobra.ShellCompDirective) {
	if len(args) == 0 {
		return ListClustersInConfig(toComplete), cobra.ShellCompDirectiveNoFileComp
	}
	return nil, cobra.ShellCompDirectiveNoFileComp
}

// ListContextsInConfig returns a list of context names which begin with `toComplete`
func ListContextsInConfig(toComplete string) []string {
	config, err := factory.ToRawKubeConfigLoader().RawConfig()
	if err != nil {
		return nil
	}
	var ret []string
	for name := range config.Contexts {
		if strings.HasPrefix(name, toComplete) {
			ret = append(ret, name)
		}
	}
	return ret
}

// ListClustersInConfig returns a list of cluster names which begin with `toComplete`
func ListClustersInConfig(toComplete string) []string {
	config, err := factory.ToRawKubeConfigLoader().RawConfig()
	if err != nil {
		return nil
	}
	var ret []string
	for name := range config.Clusters {
		if strings.HasPrefix(name, toComplete) {
			ret = append(ret, name)
		}
	}
	return ret
}

// ListUsersInConfig returns a list of user names which begin with `toComplete`
func ListUsersInConfig(toComplete string) []string {
	config, err := factory.ToRawKubeConfigLoader().RawConfig()
	if err != nil {
		return nil
	}
	var ret []string
	for name := range config.AuthInfos {
		if strings.HasPrefix(name, toComplete) {
			ret = append(ret, name)
		}
	}
	return ret
}
