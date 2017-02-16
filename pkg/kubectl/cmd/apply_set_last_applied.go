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
	"fmt"
	"io"

	"github.com/golang/glog"
	"github.com/spf13/cobra"

	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/json"
	"k8s.io/kubernetes/pkg/api/annotations"
	"k8s.io/kubernetes/pkg/kubectl"
	"k8s.io/kubernetes/pkg/kubectl/cmd/templates"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/kubectl/resource"
)

type SetLastAppliedOptions struct {
	FilenameOptions              resource.FilenameOptions
	Selector                     string
	LastAppliedConfigurationList []string
	OutputFormat                 string
	Factory                      cmdutil.Factory
	Out                          io.Writer
	ErrOut                       io.Writer
}

var (
	applySetLastAppliedLong = templates.LongDesc(`
		Set the latest last-applied-configuration annotations by file.`)

	applySetLastAppliedExample = templates.Examples(`
		# Set the current file's content to the latest-applied-configuration of this resource.
		kubectl apply set-last-applied -f deploy.yaml`)
)

func NewCmdApplySetLastApplied(f cmdutil.Factory, out, err io.Writer) *cobra.Command {
	options := &SetLastAppliedOptions{Out: out, ErrOut: err}
	cmd := &cobra.Command{
		Use:     "set-last-applied -f FILENAME",
		Short:   "Set latest last-applied-configuration annotations of by file",
		Long:    applySetLastAppliedLong,
		Example: applySetLastAppliedExample,
		Run: func(cmd *cobra.Command, args []string) {
			cmdutil.CheckErr(options.RunApplySetLastApplied(f, args, cmd, out, err, options))
		},
	}

	cmdutil.AddDryRunFlag(cmd)
	cmdutil.AddRecordFlag(cmd)
	cmdutil.AddPrinterFlags(cmd)
	usage := "that contains the last-applied-configuration annotations"
	kubectl.AddJsonFilenameFlag(cmd, &options.FilenameOptions.Filenames, "Filename, directory, or URL to files "+usage)

	return cmd
}

func (o *SetLastAppliedOptions) RunApplySetLastApplied(f cmdutil.Factory, args []string, cmd *cobra.Command, out, errOut io.Writer, options *SetLastAppliedOptions) error {
	mapper, typer, err := f.UnstructuredObject()
	if err != nil {
		return err
	}

	cmdNamespace, enforceNamespace, err := f.DefaultNamespace()
	if err != nil {
		return err
	}

	r := resource.NewBuilder(mapper, typer, resource.ClientMapperFunc(f.UnstructuredClientForMapping), unstructured.UnstructuredJSONScheme).
		NamespaceParam(cmdNamespace).DefaultNamespace().
		FilenameParam(enforceNamespace, &o.FilenameOptions).
		Latest().
		Flatten().
		Do()
	err = r.Err()
	if err != nil {
		return err
	}

	dryRun := cmdutil.GetFlagBool(cmd, "dry-run")
	output := cmdutil.GetFlagString(cmd, "output")
	shortOutput := output == "name"
	codec := f.JSONEncoder()

	err = r.Visit(func(info *resource.Info, err error) error {
		if err != nil {
			return err
		}

		objMap := map[string]map[string]map[string]string{}
		metadataMap := map[string]map[string]string{}
		annotationsMap := map[string]string{}

		localFile, err := runtime.Encode(codec, info.VersionedObject)
		if err != nil {
			return err
		}
		annotationsMap[annotations.LastAppliedConfigAnnotation] = string(localFile)
		metadataMap["annotations"] = annotationsMap
		objMap["metadata"] = metadataMap
		jsonString, err := json.Marshal(objMap)
		if err != nil {
			return err
		}

		if err := info.Get(); err != nil {
			if errors.IsNotFound(err) {
				return err
			} else {
				return cmdutil.AddSourceToErr(fmt.Sprintf("retrieving current configuration of:\n%v\nfrom server for:", info), info.Source, err)
			}
		}

		if !dryRun {
			mapping := info.ResourceMapping()
			client, err := f.UnstructuredClientForMapping(mapping)
			if err != nil {
				return err
			}
			helper := resource.NewHelper(client, mapping)

			_, err = helper.Patch(cmdNamespace, info.Name, types.MergePatchType, jsonString)
			if err != nil {
				return err
			}

			if cmdutil.ShouldRecord(cmd, info) {
				if patch, patchType, err := cmdutil.ChangeResourcePatch(info, f.Command()); err == nil {
					if _, err = helper.Patch(info.Namespace, info.Name, patchType, patch); err != nil {
						glog.V(4).Infof("error recording reason: %v", err)
					}
				}
			}
		}

		if len(output) > 0 && !shortOutput {
			info.Refresh(info.Object, false)
			return cmdutil.PrintResourceInfoForCommand(cmd, info, f, out)
		}
		cmdutil.PrintSuccess(mapper, shortOutput, out, info.Mapping.Resource, info.Name, dryRun, "configured")
		return nil
	})

	if err != nil {
		return err
	}

	return nil
}
