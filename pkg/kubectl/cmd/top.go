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

package cmd

import (
	"fmt"
	"io"
	"strings"

	"github.com/renstrom/dedent"
	"github.com/spf13/cobra"
	"k8s.io/kubernetes/pkg/kubectl"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/unversioned"
)

// TopOptions is the start of the data required to perform the operation. As new fields are added, add them here instead of
// referencing the cmd.Flags()
type TopOptions struct {
}

var (
	topLong = dedent.Dedent(`
		Display Resource (CPU/Memory/Storage) usage of nodes or pods.

		The top command allows you to see the resource consumption of the nodes or pods.
		It downloads the usage metrics of a given resource (node/pod) via the Resource Metrics API.
		`)

	topExample = dedent.Dedent(`
		  # Show metrics for all nodes in the default namespace
		  kubectl top node

		  # Show metrics for a given pod in the default namespace
		  kubectl top pod POD_NAME

		  # Show metrics for the pods defined by the selector query
		  kubectl top pod --selector="key: value"`)
)

var HandledResources []unversioned.GroupKind = []unversioned.GroupKind{
	api.Kind("Pod"),
	api.Kind("Node"),
};

var GetResourcesHandledByTop = func() []string {
	keys := make([]string, 0)
	for _, k := range HandledResources {
		resource := strings.ToLower(k.Kind)
		keys = append(keys, resource)
	}
	return keys
}

func NewCmdTop(f *cmdutil.Factory, out io.Writer) *cobra.Command {
	options := &TopOptions{}

	// retrieve a list of handled resources as valid args
	validArgs := GetResourcesHandledByTop()
	argAliases := kubectl.ResourceAliases(validArgs)

	cmd := &cobra.Command{
		Use:     "top TYPE [NAME] [flags]",
		Short:   "Display Resource (CPU/Memory/Storage) usage of nodes or pods",
		Long:    topLong,
		Example: topExample,
		Run: func(cmd *cobra.Command, args []string) {
			err := RunTop(f, out, cmd, args, options)
			cmdutil.CheckErr(err)
		},
		ValidArgs:  validArgs,
		ArgAliases: argAliases,
	}
	cmd.Flags().StringP("selector", "l", "", "Selector (label query) to filter on")
	cmd.Flags().Bool("all-namespaces", false, "If present, list the requested object(s) across all namespaces. Namespace in current context is ignored even if specified with --namespace.")
	cmd.Flags().Bool("containers", false, "If present, print usage of containers within a pod. Ignored if the requested resource type is \"node\".")
	cmdutil.AddInclude3rdPartyFlags(cmd)
	return cmd
}

func RunTop(f *cmdutil.Factory, out io.Writer, cmd *cobra.Command, args []string, options *TopOptions) error {
	if len(args) == 0 {
		fmt.Fprint(out, "You must specify the type of resource to get.")
		return cmdutil.UsageError(cmd, "Required resource not specified.")
	}

	selector := cmdutil.GetFlagString(cmd, "selector")
	allNamespaces := cmdutil.GetFlagBool(cmd, "all-namespaces")
	printContainers := cmdutil.GetFlagBool(cmd, "containers")

	if allNamespaces == true {
		fmt.Fprintf(out, "--all-namespaces not supported yet.\n")
		return cmdutil.UsageError(cmd, "Flag --all-namespaces is not supported yet.")
	}

	cmdNamespace, _, err := f.DefaultNamespace()
	if err != nil {
		return err
	}

	cli, err := f.Client()
	if err != nil {
		return err
	}

	params := map[string]string{"labelSelector": selector}

	// TODO: refactor to builder?
	err = kubectl.PrintMetrics(out, cli, cmdNamespace, allNamespaces, printContainers, params, args)
	if err != nil {
		return err
	}
	return nil
}