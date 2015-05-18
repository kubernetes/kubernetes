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
	"encoding/json"
	"io"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/kubectl"
	cmdutil "github.com/GoogleCloudPlatform/kubernetes/pkg/kubectl/cmd/util"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/kubectl/resource"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util/strategicpatch"

	"github.com/spf13/cobra"
)

const (
	diff_long = `Display any differences between the current configuration of a
resource and a configuration file.`

	diff_example = `$ kubectl diff -f my-pod.json`
)

// NewCmdDiff creates a command object for the "diff" action.
func NewCmdDiff(f *cmdutil.Factory, out io.Writer) *cobra.Command {
	var filenames util.StringList
	cmd := &cobra.Command{
		Use:     "diff (FILENAME)",
		Short:   "Show differences between the current configuration of a resource and a configuration file.",
		Long:    diff_long,
		Example: diff_example,
		Run: func(cmd *cobra.Command, args []string) {
			err := RunDiff(f, out, cmd, args, filenames)
			cmdutil.CheckErr(err)
		},
	}
	cmdutil.AddPrinterFlags(cmd)
	usage := "Filename containing the resource configuration to be compared."
	kubectl.AddJsonFilenameFlag(cmd, &filenames, usage)
	return cmd
}

func RunDiff(f *cmdutil.Factory, out io.Writer, cmd *cobra.Command, args []string, filenames util.StringList) error {
	mapper, typer := f.Object()

	cmdNamespace, enforceNamespace, err := f.DefaultNamespace()
	if err != nil {
		return err
	}

	b := resource.NewBuilder(mapper, typer, f.ClientMapperForCommand()).
		NamespaceParam(cmdNamespace).DefaultNamespace().
		FilenameParam(enforceNamespace, filenames...).
		ContinueOnError().
		Do()

	return b.Visit(func(info *resource.Info) error {
		// Get the corresponding resource that was defined in the file
		obj, err := resource.NewHelper(info.Client, info.Mapping).Get(info.Namespace, info.Name)
		if err != nil {
			return err
		}

		// Convert the objects to byte arrays
		objFileBytes, _ := json.Marshal(info.Object)
		objBytes, _ := json.Marshal(obj)

		diffString, err := strategicpatch.StrategicMergeDiff(objFileBytes, objBytes, info.Object)
		if err != nil {
			return err
		}
		out.Write(diffString)

		return nil
	})
}
