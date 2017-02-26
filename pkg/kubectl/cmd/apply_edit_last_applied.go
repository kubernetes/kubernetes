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
	"bytes"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"reflect"
	gruntime "runtime"

	"github.com/spf13/cobra"

	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/kubectl"
	"k8s.io/kubernetes/pkg/kubectl/cmd/templates"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/kubectl/cmd/util/editor"
	"k8s.io/kubernetes/pkg/kubectl/resource"
	"k8s.io/kubernetes/pkg/util/crlf"
)

type EditLastAppliedOptions struct {
	editPrinterOptions
	FilenameOptions              resource.FilenameOptions
	Selector                     string
	Namespace                    string
	LastAppliedConfigurationList []string
	OutputFormat                 string
	SchemaCacheDir               string
	file                         string
	CreateAnnotation             bool
	Factory                      cmdutil.Factory
	Codec                        runtime.Encoder
	Out                          io.Writer
	ErrOut                       io.Writer
	Mapper                       meta.RESTMapper
	EditBuffer                   *bytes.Buffer
	updatedResultsGetter         resultGetter
	diffObject                   []runtime.Object
	updatedInfos                 []*resource.Info
}

var (
	applyEditLastAppliedLong = templates.LongDesc(`
		Edit the latest last-applied-configuration annotations of resources from the default editor.

		The edit command allows you to directly edit any API resource you can retrieve via the
		command line tools. It will open the editor defined by your KUBE_EDITOR, or EDITOR
		environment variables, or fall back to 'vi' for Linux or 'notepad' for Windows.
		You can edit multiple objects, although changes are applied one at a time. The command
		accepts filenames as well as command line arguments, although the files you point to must
		be previously saved versions of resources.

		Editing is done with the API version used to fetch the resource.
		To edit using a specific API version, fully-qualify the resource, version, and group.

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

		# Edit the last-applied-configuration annotations by file in JSON
		kubectl apply edit-last-applied -f deploy.yaml -o json`)
)

func NewCmdApplyEditLastApplied(f cmdutil.Factory, out, err io.Writer) *cobra.Command {
	options := &EditLastAppliedOptions{Out: out, ErrOut: err, Factory: f}
	cmd := &cobra.Command{
		Use:     "edit-last-applied (RESOURCE/NAME | -f FILENAME)",
		Short:   "Edit latest last-applied-configuration annotations of a resource/object",
		Long:    applyEditLastAppliedLong,
		Example: applyEditLastAppliedExample,
		Run: func(cmd *cobra.Command, args []string) {
			cmdutil.CheckErr(options.Complete(cmd, args))
			cmdutil.CheckErr(options.Validate())
			cmdutil.CheckErr(options.RunEditLastApplied())
		},
	}

	usage := "to use to edit the resource"
	cmd.Flags().BoolVar(&options.CreateAnnotation, "create-annotation", false, "Will create 'last-applied-configuration' annotations if current objects doesn't have one")
	cmdutil.AddFilenameOptionFlags(cmd, &options.FilenameOptions, usage)
	cmd.Flags().StringP("output", "o", "yaml", "Output format. One of: yaml|json.")
	cmd.Flags().String("output-version", "", "DEPRECATED: To edit using a specific API version, fully-qualify the resource, version, and group (for example: 'jobs.v1.batch/myjob').")
	cmd.Flags().MarkDeprecated("output-version", "editing is now done using the resource exactly as fetched from the API. To edit using a specific API version, fully-qualify the resource, version, and group (for example: 'jobs.v1.batch/myjob').")
	cmd.Flags().MarkHidden("output-version")
	cmd.Flags().Bool("windows-line-endings", gruntime.GOOS == "windows", "Use Windows line-endings (default Unix line-endings)")
	cmdutil.AddRecordFlag(cmd)

	return cmd
}

func (o *EditLastAppliedOptions) Complete(cmd *cobra.Command, args []string) error {
	printer, err := getPrinter(cmd)
	if err != nil {
		return err
	}
	mapper, originalResult, updatedResultsGetter, cmdNamespace, err := getMapperAndResult(o.Factory, args, &o.FilenameOptions, NormalEditMode)
	if err != nil {
		return err
	}
	o.printer = printer.printer
	o.ext = printer.ext
	o.addHeader = printer.addHeader
	o.updatedResultsGetter = updatedResultsGetter
	o.Namespace = cmdNamespace
	o.OutputFormat = cmdutil.GetFlagString(cmd, "output")
	o.Codec = o.Factory.JSONEncoder()
	o.Mapper = mapper
	windowsLineEndings := cmdutil.GetFlagBool(cmd, "windows-line-endings")

	infos, err := originalResult.Infos()
	if err != nil {
		return err
	}

	// generate the file to edit
	o.EditBuffer = &bytes.Buffer{}
	var w io.Writer = o.EditBuffer
	if windowsLineEndings {
		w = crlf.NewCRLFWriter(w)
	}
	if o.addHeader {
		o.writeAnnotationsHeader(w)
	}
	l := &api.List{
		TypeMeta: metav1.TypeMeta{
			Kind:       "List",
			APIVersion: "v1",
		},
	}

	for ix := range infos {
		info := infos[ix]
		data, err := kubectl.GetOriginalConfiguration(info.Mapping, info.Object)
		if err != nil {
			return err
		}
		if data == nil && !o.CreateAnnotation {
			return cmdutil.UsageError(cmd, "no last-applied-configuration annotation found on resource: %s, to create the annotation, run the command with --create-annotation", info.Name)
		}

		getInfo, err := updatedResultsGetter(data).Infos()
		if err != nil {
			return err
		}
		if len(getInfo) != 0 {
			obj := getInfo[0].Object
			o.diffObject = append(o.diffObject, obj)
			if err != nil {
				return err
			}
			l.Items = append(l.Items, obj)
		} else {
			l.Items = append(l.Items, nil)
		}

	}
	if err := o.printer.PrintObj(l, w); err != nil {
		return preservedFile(err, o.file, o.ErrOut)
	}
	return nil
}

func (o *EditLastAppliedOptions) Validate() error {
	edit := editor.NewDefaultEditor(o.Factory.EditorEnvs())
	editedDiff := o.EditBuffer.Bytes()
	edited, file, err := edit.LaunchTempFile(fmt.Sprintf("%s-edit-", filepath.Base(os.Args[0])), o.ext, o.EditBuffer)
	o.file = file
	if err != nil {
		return preservedFile(err, o.file, o.ErrOut)
	}
	if bytes.Equal(stripComments(editedDiff), stripComments(edited)) {
		return fmt.Errorf("Edit cancelled, no changes made.")
	}

	o.updatedInfos, err = o.updatedResultsGetter(edited).Infos()
	if err != nil {
		return preservedFile(fmt.Errorf("%s", "Edit cancelled, no valid changes were saved."), o.file, o.ErrOut)
	}

	if len(o.updatedInfos) == 0 {
		return fmt.Errorf("Edit cancelled, no valid changes were saved.")
	}

	return nil
}

func (o *EditLastAppliedOptions) RunEditLastApplied() error {
	for ix := range o.updatedInfos {
		updatedInfo := o.updatedInfos[ix]
		originalJS, editedJS, err := SerializationConvert(o.Codec, o.diffObject[ix], updatedInfo.Object)
		if err != nil {
			return err
		}

		if reflect.DeepEqual(originalJS, editedJS) {
			cmdutil.PrintSuccess(o.Mapper, false, o.Out, updatedInfo.Mapping.Resource, updatedInfo.Name, false, "skipped")
		} else {
			o.annotationPatch(updatedInfo)
			cmdutil.PrintSuccess(o.Mapper, false, o.Out, updatedInfo.Mapping.Resource, updatedInfo.Name, false, "edited")
		}
	}
	return nil
}

func (o *EditLastAppliedOptions) annotationPatch(update *resource.Info) error {
	//TODO: move apply to it's own package
	p := &SetLastAppliedOptions{}
	patch, _, err := p.getPatch(update, o.Codec, false)
	if err != nil {
		return err
	}
	mapping := update.ResourceMapping()
	client, err := o.Factory.UnstructuredClientForMapping(mapping)
	if err != nil {
		return err
	}
	helper := resource.NewHelper(client, mapping)
	_, err = helper.Patch(o.Namespace, update.Name, types.MergePatchType, patch)
	if err != nil {
		return preservedFile(err, o.file, o.ErrOut)
	}
	return nil
}

func (o *EditLastAppliedOptions) writeAnnotationsHeader(w io.Writer) {
	fmt.Fprint(w, `# Please edit the 'last-applied-configuration' annotations below.
#Lines beginning with a '#' will be ignored, and an empty file will abort the edit.
#
`)
}
