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
	update_long = `Update a resource by filename or stdin.

JSON and YAML formats are accepted.`
	update_example = `// Update a pod using the data in pod.json.
$ kubectl update -f pod.json

// Update a pod based on the JSON passed into stdin.
$ cat pod.json | kubectl update -f -`
)

func NewCmdUpdate(f *cmdutil.Factory, out io.Writer) *cobra.Command {
	var filenames util.StringList
	cmd := &cobra.Command{
		Use:     "update -f FILENAME",
		Short:   "Update a resource by filename or stdin.",
		Long:    update_long,
		Example: update_example,
		Run: func(cmd *cobra.Command, args []string) {
			err := RunUpdate(f, out, cmd, args, filenames)
			cmdutil.CheckErr(err)
		},
	}
	usage := "Filename, directory, or URL to file to use to update the resource."
	kubectl.AddJsonFilenameFlag(cmd, &filenames, usage)
	cmd.MarkFlagRequired("filename")
	// TODO: re-enable --patch and make it use strategic-merge-patch
	// cmd.Flags().String("patch", "", "A JSON document to override the existing resource. The resource is downloaded, patched with the JSON, then updated.")
	// cmd.MarkFlagRequired("patch")
	return cmd
}

func RunUpdate(f *cmdutil.Factory, out io.Writer, cmd *cobra.Command, args []string, filenames util.StringList) error {
	schema, err := f.Validator()
	if err != nil {
		return err
	}

	cmdNamespace, err := f.DefaultNamespace()
	if err != nil {
		return err
	}

	/* patch := cmdutil.GetFlagString(cmd, "patch")
	if len(filenames) == 0 && len(patch) == 0 {
		return cmdutil.UsageError(cmd, "Must specify --filename or --patch to update")
	}
	if len(filenames) != 0 && len(patch) != 0 {
		return cmdutil.UsageError(cmd, "Can not specify both --filename and --patch")
	}

	// TODO: Make patching work with -f, updating with patched JSON input files
	if len(filenames) == 0 {
		name, err := updateWithPatch(cmd, args, f, patch)
		if err != nil {
			return err
		}
		fmt.Fprintf(out, "%s\n", name)
		return nil
	} */
	if len(filenames) == 0 {
		return cmdutil.UsageError(cmd, "Must specify --filename to update")
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

	return r.Visit(func(info *resource.Info) error {
		data, err := info.Mapping.Codec.Encode(info.Object)
		if err != nil {
			return err
		}
		obj, err := resource.NewHelper(info.Client, info.Mapping).Update(info.Namespace, info.Name, true, data)
		if err != nil {
			return err
		}
		info.Refresh(obj, true)
		fmt.Fprintf(out, "%s/%s\n", info.Mapping.Resource, info.Name)
		return nil
	})
}

func updateWithPatch(cmd *cobra.Command, args []string, f *cmdutil.Factory, patch string) (string, error) {
	cmdNamespace, err := f.DefaultNamespace()
	if err != nil {
		return "", err
	}

	mapper, typer := f.Object()
	r := resource.NewBuilder(mapper, typer, f.ClientMapperForCommand()).
		ContinueOnError().
		NamespaceParam(cmdNamespace).DefaultNamespace().
		ResourceTypeOrNameArgs(false, args...).
		Flatten().
		Do()
	err = r.Err()
	if err != nil {
		return "", err
	}
	mapping, err := r.ResourceMapping()
	if err != nil {
		return "", err
	}
	client, err := f.RESTClient(mapping)
	if err != nil {
		return "", err
	}

	infos, err := r.Infos()
	if err != nil {
		return "", err
	}
	name, namespace := infos[0].Name, infos[0].Namespace

	helper := resource.NewHelper(client, mapping)
	obj, err := helper.Get(namespace, name)
	if err != nil {
		return "", err
	}

	patchedObj, err := cmdutil.Merge(obj, patch, mapping.Kind)
	if err != nil {
		return "", err
	}

	data, err := helper.Codec.Encode(patchedObj)
	if err != nil {
		return "", err
	}

	_, err = helper.Update(namespace, name, true, data)
	return name, err
}
