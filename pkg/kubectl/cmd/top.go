/*
Copyright 2016 The Kubernetes Authors.

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
	"errors"
	"fmt"
	"io"
	"strings"

	"github.com/renstrom/dedent"
	"github.com/spf13/cobra"
	"k8s.io/kubernetes/pkg/kubectl"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/kubectl/metricsutil"
)

// TopOptions is the start of the data required to perform the operation. As new fields are added, add them here instead of
// referencing the cmd.Flags()
type TopOptions struct {
}

var (
	topLong = dedent.Dedent(`
		Display Resource (CPU/Memory/Storage) usage of nodes or pods.

		The top command allows you to see the resource consumption of the nodes or pods.
		It downloads the usage metrics of a given resource (node/pod) via the Resource Metrics API.`)

	topExample = dedent.Dedent(`
		  # Show metrics for all nodes
		  kubectl top node

		  # Show metrics for a given node
		  kubectl top node NODE_NAME

		  # Show metrics for all pods in the given namespace
		  kubectl top pod --namespace=NAMESPACE

		  # Show metrics for a given pod and its containers
		  kubectl top pod POD_NAME --containers

		  # Show metrics for the pods defined by label name=myLabel
		  kubectl top pod -l name=myLabel`)
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
	cmd.Flags().Bool("containers", false, "If present, print usage of containers within a pod. Ignored if the requested resource type is \"node\".")
	// TODO: add support for --all-namespaces
	cmdutil.AddInclude3rdPartyFlags(cmd)
	return cmd
}

func RunTop(f *cmdutil.Factory, out io.Writer, cmd *cobra.Command, args []string, options *TopOptions) error {
	if len(args) == 0 {
		fmt.Fprint(out, "You must specify the type of resource to get.")
		return cmdutil.UsageError(cmd, "Required resource not specified.")
	}
	selector := cmdutil.GetFlagString(cmd, "selector")
	params := map[string]string{"labelSelector": selector}
	printContainers := cmdutil.GetFlagBool(cmd, "containers")
	cmdNamespace, _, err := f.DefaultNamespace()
	if err != nil {
		return err
	}
	cli, err := f.Client()
	if err != nil {
		return err
	}

	err = PrintMetrics(out, cli, cmdNamespace, false, printContainers, params, args)
	if err != nil {
		return err
	}
	return nil
}

func PrintMetrics(out io.Writer, cli *client.Client, cmdNamespace string, allNamespaces bool,
		  printContainers bool, params map[string]string, args []string) error {
	resType, err := GetResourceKind(args[0])
	if err != nil {
		return err
	}
	name := ""
	if len(args) > 1 {
		name = args[1]
	}

	heapsterClient := metricsutil.DefaultHeapsterMetricsClient(cli)
	printer := metricsutil.NewTopCmdPrinter(out)

	switch resType {
	case api.Kind("Node"):
		metrics, err := heapsterClient.GetNodeMetrics(name, params)
		if err != nil {
			return err
		}
		return printer.PrintNodeMetrics(metrics)
	case api.Kind("Pod"):
		metrics, err := heapsterClient.GetPodMetrics(cmdNamespace, name, params)
		if err != nil {
			return err
		}
		return printer.PrintPodMetrics(metrics, printContainers)
	}
	return errors.New("resource not supported")
}

var GetResourceKind = func(resourceType string) (unversioned.GroupKind, error) {
	switch resourceType {
	case "node", "nodes":
		return api.Kind("Node"), nil
	case "pod", "pods":
		return api.Kind("Pod"), nil
	}
	return unversioned.GroupKind{}, errors.New("unknown resource requested")
}
