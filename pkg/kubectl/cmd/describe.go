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

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/meta"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/kubectl"
	cmdutil "github.com/GoogleCloudPlatform/kubernetes/pkg/kubectl/cmd/util"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/kubectl/resource"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
)

func NewCmdDescribe(f *cmdutil.Factory, out io.Writer) *cobra.Command {
	var filenames util.StringList
	cmd := &cobra.Command{
		Use:   "describe ([-f FILENAME] | RESOURCE NAME | RESOURCE/NAME)",
		Short: "Show details of a specific resource",
		Long: `Show details of a specific resource.

This command joins many API calls together to form a detailed description of a
given resource.`,
		Example: `// Describe a node
$ kubectl describe nodes kubernetes-minion-emt8.c.myproject.internal

// Describe a pod
$ kubectl describe pods/nginx

// Describe a pod using the data in pod.json.
$ kubectl describe -f pod.json

// Describe pods by label name=myLabel
$ kubectl describe po -l name=myLabel`,
		Run: func(cmd *cobra.Command, args []string) {
			err := RunDescribe(f, out, cmd, args, filenames)
			cmdutil.CheckErr(err)
		},
		ValidArgs: kubectl.DescribableResources(),
	}
	usage := "Filename, directory, or URL to a file containing the resource to describe"
	kubectl.AddJsonFilenameFlag(cmd, &filenames, usage)
	cmd.Flags().StringP("selector", "l", "", "Selector (label query) to filter on")
	return cmd
}

func RunDescribe(f *cmdutil.Factory, out io.Writer, cmd *cobra.Command, args []string, filenames util.StringList) error {
	selector := cmdutil.GetFlagString(cmd, "selector")
	cmdNamespace, enforceNamespace, err := f.DefaultNamespace()
	if err != nil {
		return err
	}

	mapper, typer := f.Object()
	r := resource.NewBuilder(mapper, typer, f.ClientMapperForCommand()).
		ContinueOnError().
		NamespaceParam(cmdNamespace).DefaultNamespace().
		FilenameParam(enforceNamespace, filenames...).
		SelectorParam(selector).
		ResourceTypeOrNameArgs(false, args...).
		Flatten().
		Do()
	err = r.Err()
	if err != nil {
		return err
	}

	count := 0
	err = r.Visit(func(info *resource.Info) error {
		count++
		mapping := info.ResourceMapping()
		describer, err := f.Describer(mapping)
		if err != nil {
			return err
		}
		s, err := describer.Describe(info.Namespace, info.Name)
		if err != nil {
			return err
		}
		fmt.Fprintf(out, "%s\n\n", s)
		return nil
	})
	if err != nil {
		return err
	}
	if count == 0 {
		return fmt.Errorf("no objects passed to describe")
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
		return fmt.Errorf("%v %q not found", rsrc, prefix)
	}
	return nil
}
