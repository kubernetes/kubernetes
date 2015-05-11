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

	"github.com/spf13/cobra"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/kubectl"
	cmdutil "github.com/GoogleCloudPlatform/kubernetes/pkg/kubectl/cmd/util"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/kubectl/resource"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
)

const (
	create_long = `Create a resource by filename or stdin.

JSON and YAML formats are accepted.`
	create_example = `// Create a pod using the data in pod.json.
$ kubectl create -f pod.json

// Create a pod based on the JSON passed into stdin.
$ cat pod.json | kubectl create -f -`
)

func NewCmdCreate(f *cmdutil.Factory, out io.Writer) *cobra.Command {
	var filenames util.StringList
	cmd := &cobra.Command{
		Use:     "create -f FILENAME",
		Short:   "Create a resource by filename or stdin",
		Long:    create_long,
		Example: create_example,
		Run: func(cmd *cobra.Command, args []string) {
			cmdutil.CheckErr(ValidateArgs(cmd, args))
			cmdutil.CheckErr(RunCreate(f, out, filenames))
		},
	}

	usage := "Filename, directory, or URL to file to use to create the resource"
	kubectl.AddJsonFilenameFlag(cmd, &filenames, usage)
	cmd.MarkFlagRequired("filename")

	return cmd
}

func ValidateArgs(cmd *cobra.Command, args []string) error {
	if len(args) != 0 {
		return cmdutil.UsageError(cmd, "Unexpected args: %v", args)
	}
	return nil
}

func RunCreate(f *cmdutil.Factory, out io.Writer, filenames util.StringList) error {
	schema, err := f.Validator()
	if err != nil {
		return err
	}

	cmdNamespace, err := f.DefaultNamespace()
	if err != nil {
		return err
	}

	mapper, typer := f.Object()
	r := resource.NewBuilder(mapper, typer, f.ClientMapperForCommand()).
		Schema(schema).
		ContinueOnError().
		NamespaceParam(cmdNamespace).RequireNamespace().
		FilenameParam(filenames...).
		Flatten().
		Do()
	err = r.Err()
	if err != nil {
		return err
	}

	count := 0
	err = r.Visit(func(info *resource.Info) error {
		data, err := info.Mapping.Codec.Encode(info.Object)
		if err != nil {
			return err
		}
		obj, err := resource.NewHelper(info.Client, info.Mapping).Create(info.Namespace, true, data)
		if err != nil {
			return err
		}
		count++
		info.Refresh(obj, true)
		fmt.Fprintf(out, "%s/%s\n", info.Mapping.Resource, info.Name)
		return nil
	})
	if err != nil {
		return err
	}
	if count == 0 {
		return fmt.Errorf("no objects passed to create")
	}
	return nil
}
