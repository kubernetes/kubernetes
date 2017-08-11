/*
Copyright 2017 The Kubernetes Authors.

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
	"io"
	"runtime"

	"github.com/spf13/cobra"

	"k8s.io/kubernetes/pkg/kubectl"
	"k8s.io/kubernetes/pkg/kubectl/cmd/templates"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/kubectl/cmd/util/editor"
	"k8s.io/kubernetes/pkg/printers"
)

var (
	applyEditLastAppliedLong = templates.LongDesc(`
		Edit the latest last-applied-configuration annotations of resources from the default editor.

		The edit-last-applied command allows you to directly edit any API resource you can retrieve via the
		command line tools. It will open the editor defined by your KUBE_EDITOR, or EDITOR
		environment variables, or fall back to 'vi' for Linux or 'notepad' for Windows.
		You can edit multiple objects, although changes are applied one at a time. The command
		accepts filenames as well as command line arguments, although the files you point to must
		be previously saved versions of resources.

		The default format is YAML. To edit in JSON, specify "-o json".

		The flag --windows-line-endings can be used to force Windows line endings,
		otherwise the default for your operating system will be used.

		In the event an error occurs while updating, a temporary file will be created on disk
		that contains your unapplied changes. The most common error when updating a resource
		is another editor changing the resource on the server. When this occurs, you will have
		to apply your changes to the newer version of the resource, or update your temporary
		saved copy to include the latest resource version.`)

	applyEditLastAppliedExample = templates.Examples(`
		# Edit the last-applied-configuration annotations by type/name in YAML.
		kubectl apply edit-last-applied deployment/nginx

		# Edit the last-applied-configuration annotations by file in JSON.
		kubectl apply edit-last-applied -f deploy.yaml -o json`)
)

func NewCmdApplyEditLastApplied(f cmdutil.Factory, out, errOut io.Writer) *cobra.Command {
	options := &editor.EditOptions{
		EditMode: editor.ApplyEditMode,
	}

	// retrieve a list of handled resources from printer as valid args
	validArgs, argAliases := []string{}, []string{}
	p, err := f.Printer(nil, printers.PrintOptions{
		ColumnLabels: []string{},
	})
	cmdutil.CheckErr(err)
	if p != nil {
		validArgs = p.HandledResources()
		argAliases = kubectl.ResourceAliases(validArgs)
	}

	cmd := &cobra.Command{
		Use:     "edit-last-applied (RESOURCE/NAME | -f FILENAME)",
		Short:   "Edit latest last-applied-configuration annotations of a resource/object",
		Long:    applyEditLastAppliedLong,
		Example: applyEditLastAppliedExample,
		Run: func(cmd *cobra.Command, args []string) {
			options.ChangeCause = f.Command(cmd, false)
			if err := options.Complete(f, out, errOut, args, cmd); err != nil {
				cmdutil.CheckErr(err)
			}
			if err := options.Run(); err != nil {
				cmdutil.CheckErr(err)
			}
		},
		ValidArgs:  validArgs,
		ArgAliases: argAliases,
	}

	usage := "to use to edit the resource"
	cmdutil.AddFilenameOptionFlags(cmd, &options.FilenameOptions, usage)
	cmd.Flags().StringVarP(&options.Output, "output", "o", "yaml", "Output format. One of: yaml|json.")
	cmd.Flags().BoolVar(&options.WindowsLineEndings, "windows-line-endings", runtime.GOOS == "windows",
		"Defaults to the line ending native to your platform.")
	cmdutil.AddRecordVarFlag(cmd, &options.Record)
	cmdutil.AddIncludeUninitializedFlag(cmd)

	return cmd
}
