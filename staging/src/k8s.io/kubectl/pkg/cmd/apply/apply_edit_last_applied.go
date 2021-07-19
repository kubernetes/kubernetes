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

package apply

import (
	"github.com/spf13/cobra"

	"k8s.io/cli-runtime/pkg/genericclioptions"
	cmdutil "k8s.io/kubectl/pkg/cmd/util"
	"k8s.io/kubectl/pkg/cmd/util/editor"
	"k8s.io/kubectl/pkg/util"
	"k8s.io/kubectl/pkg/util/i18n"
	"k8s.io/kubectl/pkg/util/templates"
)

var (
	applyEditLastAppliedLong = templates.LongDesc(i18n.T(`
		Edit the latest last-applied-configuration annotations of resources from the default editor.

		The edit-last-applied command allows you to directly edit any API resource you can retrieve via the
		command-line tools. It will open the editor defined by your KUBE_EDITOR, or EDITOR
		environment variables, or fall back to 'vi' for Linux or 'notepad' for Windows.
		You can edit multiple objects, although changes are applied one at a time. The command
		accepts file names as well as command-line arguments, although the files you point to must
		be previously saved versions of resources.

		The default format is YAML. To edit in JSON, specify "-o json".

		The flag --windows-line-endings can be used to force Windows line endings,
		otherwise the default for your operating system will be used.

		In the event an error occurs while updating, a temporary file will be created on disk
		that contains your unapplied changes. The most common error when updating a resource
		is another editor changing the resource on the server. When this occurs, you will have
		to apply your changes to the newer version of the resource, or update your temporary
		saved copy to include the latest resource version.`))

	applyEditLastAppliedExample = templates.Examples(`
		# Edit the last-applied-configuration annotations by type/name in YAML
		kubectl apply edit-last-applied deployment/nginx

		# Edit the last-applied-configuration annotations by file in JSON
		kubectl apply edit-last-applied -f deploy.yaml -o json`)
)

// NewCmdApplyEditLastApplied created the cobra CLI command for the `apply edit-last-applied` command.
func NewCmdApplyEditLastApplied(f cmdutil.Factory, ioStreams genericclioptions.IOStreams) *cobra.Command {
	o := editor.NewEditOptions(editor.ApplyEditMode, ioStreams)

	cmd := &cobra.Command{
		Use:                   "edit-last-applied (RESOURCE/NAME | -f FILENAME)",
		DisableFlagsInUseLine: true,
		Short:                 i18n.T("Edit latest last-applied-configuration annotations of a resource/object"),
		Long:                  applyEditLastAppliedLong,
		Example:               applyEditLastAppliedExample,
		ValidArgsFunction:     util.ResourceTypeAndNameCompletionFunc(f),
		Run: func(cmd *cobra.Command, args []string) {
			cmdutil.CheckErr(o.Complete(f, args, cmd))
			cmdutil.CheckErr(o.Run())
		},
	}

	// bind flag structs
	o.RecordFlags.AddFlags(cmd)
	o.PrintFlags.AddFlags(cmd)

	usage := "to use to edit the resource"
	cmdutil.AddFilenameOptionFlags(cmd, &o.FilenameOptions, usage)
	cmd.Flags().BoolVar(&o.WindowsLineEndings, "windows-line-endings", o.WindowsLineEndings,
		"Defaults to the line ending native to your platform.")
	cmdutil.AddFieldManagerFlagVar(cmd, &o.FieldManager, FieldManagerClientSideApply)

	return cmd
}
