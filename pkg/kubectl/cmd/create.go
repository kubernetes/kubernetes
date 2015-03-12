/*
Copyright 2014 Google Inc. All rights reserved.

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

func (f *Factory) NewCmdCreate(out io.Writer) *cobra.Command {
	var filenames util.StringList
	cmd := &cobra.Command{
		Use:     "create -f FILENAME",
		Short:   "Create a resource by filename or stdin",
		Long:    create_long,
		Example: create_example,
		Run: func(cmd *cobra.Command, args []string) {
			err := RunCreate(f, out, cmd, filenames)
			cmdutil.CheckErr(err)
		},
	}
	cmd.Flags().VarP(&filenames, "filename", "f", "Filename, directory, or URL to file to use to create the resource")
	return cmd
}

func RunCreate(f *Factory, out io.Writer, cmd *cobra.Command, filenames util.StringList) error {
	schema, err := f.Validator(cmd)
	if err != nil {
		return err
	}

	cmdNamespace, err := f.DefaultNamespace(cmd)
	if err != nil {
		return err
	}

	mapper, typer := f.Object(cmd)
	r := resource.NewBuilder(mapper, typer, f.ClientMapperForCommand(cmd)).
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
		if err := schema.ValidateBytes(data); err != nil {
			return err
		}
		obj, err := resource.NewHelper(info.Client, info.Mapping).Create(info.Namespace, true, data)
		if err != nil {
			return err
		}
		count++
		info.Refresh(obj, true)
		fmt.Fprintf(out, "%s\n", info.Name)
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
