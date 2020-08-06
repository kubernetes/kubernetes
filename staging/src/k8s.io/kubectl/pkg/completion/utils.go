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

package completion

import (
	"context"
	"strings"
	"time"

	"github.com/spf13/cobra"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubectl/pkg/cmd/util"
)

// ListNamespaces returns a list of namespaces which begins with `toComplete`.
func ListNamespaces(f util.Factory, toComplete string) ([]string, cobra.ShellCompDirective) {
	clientSet, err := f.KubernetesClientSet()
	if err != nil {
		return nil, cobra.ShellCompDirectiveDefault
	}
	ctx, cancel := context.WithTimeout(context.TODO(), time.Second*3)
	defer cancel()
	namespaces, err := clientSet.CoreV1().Namespaces().List(ctx, metav1.ListOptions{})
	if err != nil {
		return nil, cobra.ShellCompDirectiveDefault
	}
	var ret []string
	for _, ns := range namespaces.Items {
		if strings.HasPrefix(ns.Name, toComplete) {
			ret = append(ret, ns.Name)
		}
	}
	return ret, cobra.ShellCompDirectiveNoFileComp
}

// ListContextsInKubeconfig returns a list of context names which begins with `toComplete`.
func ListContextsInKubeconfig(f util.Factory, toComplete string) ([]string, cobra.ShellCompDirective) {
	config, err := f.ToRawKubeConfigLoader().RawConfig()
	if err != nil {
		return nil, cobra.ShellCompDirectiveNoFileComp
	}
	var ret []string
	for name := range config.Contexts {
		if strings.HasPrefix(name, toComplete) {
			ret = append(ret, name)
		}
	}
	return ret, cobra.ShellCompDirectiveNoFileComp
}

// ListClustersInKubeconfig returns a list of cluster names which begins with `toComplete`.
func ListClustersInKubeconfig(f util.Factory, toComplete string) ([]string, cobra.ShellCompDirective) {
	config, err := f.ToRawKubeConfigLoader().RawConfig()
	if err != nil {
		return nil, cobra.ShellCompDirectiveNoFileComp
	}
	var ret []string
	for name := range config.Clusters {
		if strings.HasPrefix(name, toComplete) {
			ret = append(ret, name)
		}
	}
	return ret, cobra.ShellCompDirectiveNoFileComp
}

// ListUsersInKubeconfig returns a list of user names which begins with `toComplete`.
func ListUsersInKubeconfig(f util.Factory, toComplete string) ([]string, cobra.ShellCompDirective) {
	config, err := f.ToRawKubeConfigLoader().RawConfig()
	if err != nil {
		return nil, cobra.ShellCompDirectiveNoFileComp
	}
	var ret []string
	for name := range config.AuthInfos {
		if strings.HasPrefix(name, toComplete) {
			ret = append(ret, name)
		}
	}
	return ret, cobra.ShellCompDirectiveNoFileComp
}
