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
	"encoding/json"
	"fmt"
	"io"

	"github.com/ghodss/yaml"
	"github.com/spf13/cobra"

	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	apijson "k8s.io/apimachinery/pkg/util/json"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/kubectl"
	"k8s.io/kubernetes/pkg/kubectl/cmd/templates"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/kubectl/cmd/util/editor"
	"k8s.io/kubernetes/pkg/kubectl/resource"
	"k8s.io/kubernetes/pkg/kubectl/util/i18n"
)

type SetLastAppliedOptions struct {
	FilenameOptions  resource.FilenameOptions
	Selector         string
	InfoList         []*resource.Info
	Mapper           meta.RESTMapper
	Typer            runtime.ObjectTyper
	Namespace        string
	EnforceNamespace bool
	DryRun           bool
	ShortOutput      bool
	CreateAnnotation bool
	Output           string
	Codec            runtime.Encoder
	PatchBufferList  []PatchBuffer
	Factory          cmdutil.Factory
	Out              io.Writer
	ErrOut           io.Writer
}

type PatchBuffer struct {
	Patch     []byte
	PatchType types.PatchType
}

var (
	applySetLastAppliedLong = templates.LongDesc(i18n.T(`
		Set the latest last-applied-configuration annotations by setting it to match the contents of a file.
		This results in the last-applied-configuration being updated as though 'kubectl apply -f <file>' was run,
		without updating any other parts of the object.`))

	applySetLastAppliedExample = templates.Examples(i18n.T(`
		# Set the last-applied-configuration of a resource to match the contents of a file.
		kubectl apply set-last-applied -f deploy.yaml

		# Execute set-last-applied against each configuration file in a directory.
		kubectl apply set-last-applied -f path/

		# Set the last-applied-configuration of a resource to match the contents of a file, will create the annotation if it does not already exist.
		kubectl apply set-last-applied -f deploy.yaml --create-annotation=true
		`))
)

func NewCmdApplySetLastApplied(f cmdutil.Factory, out, err io.Writer) *cobra.Command {
	options := &SetLastAppliedOptions{Out: out, ErrOut: err}
	cmd := &cobra.Command{
		Use:     "set-last-applied -f FILENAME",
		Short:   i18n.T("Set the last-applied-configuration annotation on a live object to match the contents of a file."),
		Long:    applySetLastAppliedLong,
		Example: applySetLastAppliedExample,
		Run: func(cmd *cobra.Command, args []string) {
			cmdutil.CheckErr(options.Complete(f, cmd))
			cmdutil.CheckErr(options.Validate(f, cmd))
			cmdutil.CheckErr(options.RunSetLastApplied(f, cmd))
		},
	}

	cmdutil.AddDryRunFlag(cmd)
	cmdutil.AddRecordFlag(cmd)
	cmdutil.AddPrinterFlags(cmd)
	cmd.Flags().BoolVar(&options.CreateAnnotation, "create-annotation", false, "Will create 'last-applied-configuration' annotations if current objects doesn't have one")
	usage := "that contains the last-applied-configuration annotations"
	kubectl.AddJsonFilenameFlag(cmd, &options.FilenameOptions.Filenames, "Filename, directory, or URL to files "+usage)

	return cmd
}

func (o *SetLastAppliedOptions) Complete(f cmdutil.Factory, cmd *cobra.Command) error {
	o.DryRun = cmdutil.GetFlagBool(cmd, "dry-run")
	o.Output = cmdutil.GetFlagString(cmd, "output")
	o.ShortOutput = o.Output == "name"
	o.Codec = f.JSONEncoder()

	var err error
	o.Mapper, o.Typer, err = f.UnstructuredObject()
	if err != nil {
		return err
	}

	o.Namespace, o.EnforceNamespace, err = f.DefaultNamespace()
	return err
}

func (o *SetLastAppliedOptions) Validate(f cmdutil.Factory, cmd *cobra.Command) error {
	builder, err := f.NewUnstructuredBuilder(true)
	if err != nil {
		return err
	}

	r := builder.
		NamespaceParam(o.Namespace).DefaultNamespace().
		FilenameParam(o.EnforceNamespace, &o.FilenameOptions).
		Latest().
		Flatten().
		Do()
	err = r.Err()
	if err != nil {
		return err
	}

	err = r.Visit(func(info *resource.Info, err error) error {
		if err != nil {
			return err
		}

		patchBuf, diffBuf, patchType, err := editor.GetApplyPatch(info.VersionedObject, o.Codec)
		if err != nil {
			return err
		}

		// Verify the object exists in the cluster before trying to patch it.
		if err := info.Get(); err != nil {
			if errors.IsNotFound(err) {
				return err
			} else {
				return cmdutil.AddSourceToErr(fmt.Sprintf("retrieving current configuration of:\n%v\nfrom server for:", info), info.Source, err)
			}
		}
		oringalBuf, err := kubectl.GetOriginalConfiguration(info.Mapping, info.Object)
		if err != nil {
			return cmdutil.AddSourceToErr(fmt.Sprintf("retrieving current configuration of:\n%v\nfrom server for:", info), info.Source, err)
		}
		if oringalBuf == nil && !o.CreateAnnotation {
			return cmdutil.UsageErrorf(cmd, "no last-applied-configuration annotation found on resource: %s, to create the annotation, run the command with --create-annotation", info.Name)
		}

		//only add to PatchBufferList when changed
		if !bytes.Equal(cmdutil.StripComments(oringalBuf), cmdutil.StripComments(diffBuf)) {
			p := PatchBuffer{Patch: patchBuf, PatchType: patchType}
			o.PatchBufferList = append(o.PatchBufferList, p)
			o.InfoList = append(o.InfoList, info)
		} else {
			fmt.Fprintf(o.Out, "set-last-applied %s: no changes required.\n", info.Name)
		}

		return nil
	})
	return err
}

func (o *SetLastAppliedOptions) RunSetLastApplied(f cmdutil.Factory, cmd *cobra.Command) error {
	for i, patch := range o.PatchBufferList {
		info := o.InfoList[i]
		if !o.DryRun {
			mapping := info.ResourceMapping()
			client, err := f.UnstructuredClientForMapping(mapping)
			if err != nil {
				return err
			}
			helper := resource.NewHelper(client, mapping)
			patchedObj, err := helper.Patch(o.Namespace, info.Name, patch.PatchType, patch.Patch)
			if err != nil {
				return err
			}

			if len(o.Output) > 0 && !o.ShortOutput {
				info.Refresh(patchedObj, false)
				return cmdutil.PrintResourceInfoForCommand(cmd, info, f, o.Out)
			}
			cmdutil.PrintSuccess(o.Mapper, o.ShortOutput, o.Out, info.Mapping.Resource, info.Name, o.DryRun, "configured")

		} else {
			err := o.formatPrinter(o.Output, patch.Patch, o.Out)
			if err != nil {
				return err
			}
			cmdutil.PrintSuccess(o.Mapper, o.ShortOutput, o.Out, info.Mapping.Resource, info.Name, o.DryRun, "configured")
		}
	}
	return nil
}

func (o *SetLastAppliedOptions) formatPrinter(output string, buf []byte, w io.Writer) error {
	yamlOutput, err := yaml.JSONToYAML(buf)
	if err != nil {
		return err
	}
	switch output {
	case "json":
		jsonBuffer := &bytes.Buffer{}
		err = json.Indent(jsonBuffer, buf, "", "  ")
		if err != nil {
			return err
		}
		fmt.Fprintf(w, "%s\n", jsonBuffer.String())
	case "yaml":
		fmt.Fprintf(w, "%s\n", string(yamlOutput))
	}
	return nil
}

func (o *SetLastAppliedOptions) getPatch(info *resource.Info) ([]byte, []byte, error) {
	objMap := map[string]map[string]map[string]string{}
	metadataMap := map[string]map[string]string{}
	annotationsMap := map[string]string{}
	localFile, err := runtime.Encode(o.Codec, info.VersionedObject)
	if err != nil {
		return nil, localFile, err
	}
	annotationsMap[api.LastAppliedConfigAnnotation] = string(localFile)
	metadataMap["annotations"] = annotationsMap
	objMap["metadata"] = metadataMap
	jsonString, err := apijson.Marshal(objMap)
	return jsonString, localFile, err
}
