/*
Copyright 2015 The Kubernetes Authors.

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
	gruntime "runtime"

	"k8s.io/kubernetes/pkg/kubectl"
	"k8s.io/kubernetes/pkg/kubectl/cmd/templates"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/kubectl/cmd/util/editor"

	"github.com/spf13/cobra"
)

var (
	editLong = templates.LongDesc(`
		Edit a resource from the default editor.

		The edit command allows you to directly edit any API resource you can retrieve via the
		command line tools. It will open the editor defined by your KUBE_EDITOR, or EDITOR
		environment variables, or fall back to 'vi' for Linux or 'notepad' for Windows.
		You can edit multiple objects, although changes are applied one at a time. The command
		accepts filenames as well as command line arguments, although the files you point to must
		be previously saved versions of resources.

		The files to edit will be output in the default API version, or a version specified
		by --output-version. The default format is YAML - if you would like to edit in JSON
		pass -o json. The flag --windows-line-endings can be used to force Windows line endings,
		otherwise the default for your operating system will be used.

		In the event an error occurs while updating, a temporary file will be created on disk
		that contains your unapplied changes. The most common error when updating a resource
		is another editor changing the resource on the server. When this occurs, you will have
		to apply your changes to the newer version of the resource, or update your temporary
		saved copy to include the latest resource version.`)

	editExample = templates.Examples(`
		# Edit the service named 'docker-registry':
		kubectl edit svc/docker-registry

		# Use an alternative editor
		KUBE_EDITOR="nano" kubectl edit svc/docker-registry

		# Edit the service 'docker-registry' in JSON using the v1 API format:
		kubectl edit svc/docker-registry --output-version=v1 -o json`)
)

func NewCmdEdit(f cmdutil.Factory, out, errOut io.Writer) *cobra.Command {
	options := &editor.EditOptions{
		EditMode: editor.NormalEditMode,
	}

	// retrieve a list of handled resources from printer as valid args
	validArgs, argAliases := []string{}, []string{}
	p, err := f.Printer(nil, kubectl.PrintOptions{
		ColumnLabels: []string{},
	})
	cmdutil.CheckErr(err)
	if p != nil {
		validArgs = p.HandledResources()
		argAliases = kubectl.ResourceAliases(validArgs)
	}

	cmd := &cobra.Command{
		Use:     "edit (RESOURCE/NAME | -f FILENAME)",
		Short:   "Edit a resource on the server",
		Long:    editLong,
		Example: fmt.Sprintf(editExample),
		Run: func(cmd *cobra.Command, args []string) {
			cmdutil.CheckErr(cmdutil.ValidateOutputArgsForEdit(cmd))
			if err := options.Complete(f, out, errOut, args); err != nil {
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
	cmdutil.AddValidateOptionFlags(cmd, &options.ValidateOptions)
	cmd.Flags().StringVarP(&options.Output, "output", "o", "yaml", "Output format. One of: yaml|json.")
	cmd.Flags().StringVar(&options.OutputVersion, "output-version", "", "Output the formatted object with the given group version (for ex: 'extensions/v1beta1').")
	cmd.Flags().BoolVar(&options.WindowsLineEndings, "windows-line-endings", gruntime.GOOS == "windows", "Use Windows line-endings (default Unix line-endings)")
	cmdutil.AddApplyAnnotationVarFlags(cmd, &options.ApplyAnnotation)
	cmdutil.AddRecordVarFlag(cmd, &options.Record)
	cmdutil.AddInclude3rdPartyVarFlags(cmd, &options.Include3rdParty)
	return cmd
}
