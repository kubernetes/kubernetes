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

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/errors"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/meta"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/kubectl"
	cmdutil "github.com/GoogleCloudPlatform/kubernetes/pkg/kubectl/cmd/util"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/kubectl/resource"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
)

func NewCmdDescribe(f *cmdutil.Factory, out io.Writer) *cobra.Command {
	cmd := &cobra.Command{
		Use:   "describe (RESOURCE NAME | RESOURCE/NAME)",
		Short: "Show details of a specific resource",
		Long: `Show details of a specific resource.

This command joins many API calls together to form a detailed description of a
given resource.`,
		Example: `// Describe a node
$ kubectl describe nodes kubernetes-minion-emt8.c.myproject.internal

// Describe a pod
$ kubectl describe pods/nginx

// Describe pods by label name=myLabel
$ kubectl describe po -l name=myLabel`,
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
	cmdNamespace, err := f.DefaultNamespace()
	if err != nil {
		return err
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
		if errors.IsNotFound(err) && len(args) == 2 {
			return DescribeMatchingResources(mapper, typer, describer, f, cmdNamespace, args[0], args[1], out)
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

func DescribeMatchingResources(mapper meta.RESTMapper, typer runtime.ObjectTyper, describer kubectl.Describer, f *cmdutil.Factory, namespace, rsrc, prefix string, out io.Writer) error {
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
	for ix := range infos {
		info := infos[ix]
		if strings.HasPrefix(info.Name, prefix) {
			s, err := describer.Describe(info.Namespace, info.Name)
			if err != nil {
				return err
			}
			fmt.Fprintf(out, "%s\n", s)
		}
	}
	return nil
}
