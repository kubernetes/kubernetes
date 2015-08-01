/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

	"github.com/spf13/cobra"

	apierrors "github.com/GoogleCloudPlatform/kubernetes/pkg/api/errors"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/meta"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/kubectl"
	cmdutil "github.com/GoogleCloudPlatform/kubernetes/pkg/kubectl/cmd/util"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/kubectl/resource"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
)

const (
	describe_long = `Show details of a specific resource or group of resources.

This command joins many API calls together to form a detailed description of a
given resource or group of resources.

$ kubectl describe RESOURCE NAME_PREFIX

will first check for an exact match on RESOURCE and NAME_PREFIX. If no such resource
exists, it will output details for every resource that has a name prefixed with NAME_PREFIX

Possible resources include (case insensitive): pods (po), services (svc),
replicationcontrollers (rc), nodes (no), events (ev), limitranges (limits),
persistentvolumes (pv), persistentvolumeclaims (pvc), resourcequotas (quota),
namespaces (ns) or secrets.`
	describe_example = `// Describe a node
$ kubectl describe nodes kubernetes-minion-emt8.c.myproject.internal

// Describe a pod
$ kubectl describe pods/nginx

// Describe pods by label name=myLabel
$ kubectl describe po -l name=myLabel

// Describe all pods managed by the 'frontend' replication controller (rc-created pods
// get the name of the rc as a prefix in the pod the name).
$ kubectl describe pods frontend`
)

func NewCmdDescribe(f *cmdutil.Factory, out io.Writer) *cobra.Command {
	cmd := &cobra.Command{
		Use:     "describe (RESOURCE NAME_PREFIX | RESOURCE/NAME)",
		Short:   "Show details of a specific resource or group of resources",
		Long:    describe_long,
		Example: describe_example,
		Run: func(cmd *cobra.Command, args []string) {
			err := RunDescribe(f, out, cmd, args)
			cmdutil.CheckErr(err)
		},
		ValidArgs: kubectl.DescribableResources(),
	}
	cmd.Flags().StringP("selector", "l", "", "Selector (label query) to filter on")
	return cmd
}

func RunDescribe(f *cmdutil.Factory, out io.Writer, cmd *cobra.Command, args []string) error {
	selector := cmdutil.GetFlagString(cmd, "selector")
	cmdNamespace, _, err := f.DefaultNamespace()
	if err != nil {
		return err
	}

	if len(args) == 0 {
		fmt.Fprint(out, "You must specify the type of resource to describe. ", valid_resources)
		return cmdutil.UsageError(cmd, "Required resource not specified.")
	}

	mapper, typer := f.Object()
	r := resource.NewBuilder(mapper, typer, f.ClientMapperForCommand()).
		ContinueOnError().
		NamespaceParam(cmdNamespace).DefaultNamespace().
		SelectorParam(selector).
		ResourceTypeOrNameArgs(false, args...).
		Flatten().
		Do()
	err = r.Err()
	if err != nil {
		return err
	}
	mapping, err := r.ResourceMapping()
	if err != nil {
		return err
	}

	describer, err := f.Describer(mapping)
	if err != nil {
		return err
	}
	infos, err := r.Infos()
	if err != nil {
		if apierrors.IsNotFound(err) && len(args) == 2 {
			return DescribeMatchingResources(mapper, typer, describer, f, cmdNamespace, args[0], args[1], out, err)
		}
		return err
	}

	for _, info := range infos {
		s, err := describer.Describe(info.Namespace, info.Name)
		if err != nil {
			return err
		}
		fmt.Fprintf(out, "%s\n\n", s)
	}

	return nil
}

func DescribeMatchingResources(mapper meta.RESTMapper, typer runtime.ObjectTyper, describer kubectl.Describer, f *cmdutil.Factory, namespace, rsrc, prefix string, out io.Writer, originalError error) error {
	r := resource.NewBuilder(mapper, typer, f.ClientMapperForCommand()).
		NamespaceParam(namespace).DefaultNamespace().
		ResourceTypeOrNameArgs(true, rsrc).
		SingleResourceType().
		Flatten().
		Do()
	infos, err := r.Infos()
	if err != nil {
		return err
	}
	isFound := false
	for ix := range infos {
		info := infos[ix]
		if strings.HasPrefix(info.Name, prefix) {
			isFound = true
			s, err := describer.Describe(info.Namespace, info.Name)
			if err != nil {
				return err
			}
			fmt.Fprintf(out, "%s\n", s)
		}
	}
	if !isFound {
		return originalError
	}
	return nil
}
