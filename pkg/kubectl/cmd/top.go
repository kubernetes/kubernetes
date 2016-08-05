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
	"io"
	"strings"

	"github.com/renstrom/dedent"
	"github.com/spf13/cobra"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/kubectl"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/kubectl/metricsutil"
)

// TopOptions contains all the options for running the top cli command.
type TopOptions struct {
	ResourceType    unversioned.GroupKind
	ResourceName    string
	AllNamespaces   bool
	PrintContainers bool
	Selector        string
	Namespace       string
	Client          *metricsutil.HeapsterMetricsClient
	Printer         *metricsutil.TopCmdPrinter
}

var (
	topLong = dedent.Dedent(`
		Display Resource (CPU/Memory/Storage) usage of nodes or pods.

		The top command allows you to see the resource consumption of the nodes or pods.
		It downloads the usage metrics of a given resource (node/pod) via the Resource Metrics API.`)

	topExample = dedent.Dedent(`
		  # Show metrics for all nodes
		  kubectl top nodes

		  # Show metrics for a given node
		  kubectl top node NODE_NAME

		  # Show metrics for all pods in the given namespace
		  kubectl top pods --namespace=NAMESPACE

		  # Show metrics for a given pod and its containers
		  kubectl top pod POD_NAME --containers

		  # Show metrics for the pods defined by label name=myLabel
		  kubectl top pods -l name=myLabel`)
)

var HandledResources []unversioned.GroupKind = []unversioned.GroupKind{
	api.Kind("Node"),
	api.Kind("Pod"),
}

func GetResourcesHandledByTop() []string {
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
		Use:     "top TYPE [NAME | -l label]",
		Short:   "Display Resource (CPU/Memory/Storage) usage of nodes or pods",
		Long:    topLong,
		Example: topExample,
		Run: func(cmd *cobra.Command, args []string) {
			if err := options.Complete(f, cmd, args, out); err != nil {
				cmdutil.CheckErr(err)
			}
			if err := options.Validate(); err != nil {
				cmdutil.CheckErr(cmdutil.UsageError(cmd, err.Error()))
			}
			if err := options.RunTop(); err != nil {
				cmdutil.CheckErr(err)
			}
		},
		ValidArgs:  validArgs,
		ArgAliases: argAliases,
	}
	cmd.Flags().StringVarP(&options.Selector, "selector", "l", "", "Selector (label query) to filter on")
	cmd.Flags().BoolVar(&options.PrintContainers, "containers", false, "If present, print usage of containers within a pod. Ignored if the requested resource type is \"node\".")
	cmd.Flags().BoolVar(&options.AllNamespaces, "all-namespaces", false, "If present, list the requested object(s) across all namespaces. Namespace in current context is ignored even if specified with --namespace.")
	cmdutil.AddInclude3rdPartyFlags(cmd)
	return cmd
}

// Complete completes all the required options for top.
func (o *TopOptions) Complete(f *cmdutil.Factory, cmd *cobra.Command, args []string, out io.Writer) error {
	var err error
	switch len(args) {
	case 0:
		return cmdutil.UsageError(cmd, "resource name is required")
	case 1:
		o.ResourceType, err = GetResourceKind(args[0])
		if err != nil {
			return err
		}
	case 2:
		o.ResourceType, err = GetResourceKind(args[0])
		if err != nil {
			return err
		}
		o.ResourceName = args[1]
	default:
		return cmdutil.UsageError(cmd, cmd.Use)
	}

	if o.ResourceType == api.Kind("Node") && o.PrintContainers {
		return cmdutil.UsageError(cmd, "flag --containers is not allowed for nodes")
	}

	o.Namespace, _, err = f.DefaultNamespace()
	if err != nil {
		return err
	}
	cli, err := f.Client()
	if err != nil {
		return err
	}
	o.Client = metricsutil.DefaultHeapsterMetricsClient(cli)
	o.Printer = metricsutil.NewTopCmdPrinter(out)
	return nil
}

// Validate validates all the required options for top.
func (o TopOptions) Validate() error {
	err := validateHandledResourceType(o.ResourceType)
	if err != nil {
		return err
	}
	return nil
}

// RunTop implements all the necessary functionality for top.
func (o TopOptions) RunTop() error {
	switch o.ResourceType {
	case api.Kind("Node"):
		metrics, err := o.Client.GetNodeMetrics(o.ResourceName, o.Selector)
		if err != nil {
			return err
		}
		return o.Printer.PrintNodeMetrics(metrics)
	case api.Kind("Pod"):
		metrics, err := o.Client.GetPodMetrics(o.Namespace, o.ResourceName, o.AllNamespaces, o.Selector)
		if err != nil {
			return err
		}
		return o.Printer.PrintPodMetrics(metrics, o.PrintContainers)
	}
	return errors.New("unsupported resource requested")
}

func GetResourceKind(resourceType string) (unversioned.GroupKind, error) {
	if kind, found := AliasesToKind[resourceType]; found {
		return kind, nil
	}
	return unversioned.GroupKind{}, errors.New("unsupported resource requested")
}

var AliasesToKind map[string]unversioned.GroupKind = GetAliasesToKind(HandledResources)

func GetAliasesToKind(resources []unversioned.GroupKind) map[string]unversioned.GroupKind {
	aliasesToKind := make(map[string]unversioned.GroupKind, 0)
	for _, res := range resources {
		resName := strings.ToLower(res.Kind)
		aliasesToKind[resName] = res
		for _, alias := range kubectl.ResourceAliases([]string{resName}) {
			aliasesToKind[alias] = res
		}
	}
	return aliasesToKind
}

func validateHandledResourceType(resourceType unversioned.GroupKind) error {
	for _, r := range HandledResources {
		if r == resourceType {
			return nil
		}
	}
	return errors.New("unsupported resource requested")
}
