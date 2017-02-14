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
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/kubernetes/pkg/api/annotations"
	"k8s.io/kubernetes/pkg/kubectl"
	"k8s.io/kubernetes/pkg/kubectl/cmd/templates"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/kubectl/resource"
)

var (
	applyViewLastAppliedLong = templates.LongDesc(`
		Get the last-applied-configuration annotations by type/name or file.

		You can use --output jsonpath={...} to extract specific values using a jsonpath expression.`)

	applyViewLastAppliedExample = templates.Examples(`
		# Get the last applied configuration file for a resource by type/name.
		# This is retrieved from the last-applied-configuration annotation on the resource
		kubectl apply view last-applied deployment/nginx

		# Get the last-applied-configuration annotations by file
		kubectl apply view last-applied -f deployment_nginx.yaml -o json`)
)

func NewCmdApplyViewLastApplied(f cmdutil.Factory, out, err io.Writer) *cobra.Command {
	options := &ApplyOptions{}
	cmd := &cobra.Command{
		Use:     "last-applied (TYPE [NAME | -l label] | TYPE/NAME | -f FILENAME)",
		Short:   "Get the last-applied-configuration annotations",
		Long:    applyViewLastAppliedLong,
		Example: applyViewLastAppliedExample,
		Run: func(cmd *cobra.Command, args []string) {
			err := RunApplyViewLastApplied(f, out, err, cmd, args, options)
			cmdutil.CheckErr(err)
		},
	}

	cmdutil.AddPrinterFlags(cmd)
	cmd.Flags().StringVarP(&options.Selector, "selector", "l", "", "Selector (label query) to filter on, supports '=', '==', and '!='.")
	usage := "that contains the last-applied-configuration annotations"
	cmdutil.AddFilenameOptionFlags(cmd, &options.FilenameOptions, usage)

	return cmd
}

func RunApplyViewLastApplied(f cmdutil.Factory, cmdOut, errOut io.Writer, cmd *cobra.Command, args []string, options *ApplyOptions) error {
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
		FilenameParam(enforceNamespace, &options.FilenameOptions).
		ResourceTypeOrNameArgs(enforceNamespace, args...).
		SelectorParam(options.Selector).
		Latest().
		Flatten().
		Do()
	err = r.Err()
	if err != nil {
		return err
	}

	outputFormat := cmdutil.GetFlagString(cmd, "output")

	err = r.Visit(func(info *resource.Info, err error) error {
		if err != nil {
			return err
		}

		annotationMap, err := info.Mapping.MetadataAccessor.Annotations(info.Object)
		if err != nil {
			return err
		}

		if annotationMap[annotations.LastAppliedConfigAnnotation] == "" {
			fmt.Fprintf(errOut, "no last-applied-configuration annotation found on resource")
			return nil
		}

		configString := []byte(annotationMap[annotations.LastAppliedConfigAnnotation])
		yamlString, err := yaml.JSONToYAML(configString)
		if err != nil {
			return err
		}
		switch outputFormat {
		case "yaml", "":
			fmt.Fprintf(cmdOut, string(yamlString))
		case "json":
			jsonBuffer := &bytes.Buffer{}
			err = json.Indent(jsonBuffer, configString, "", "  ")
			if err != nil {
				return err
			}
			fmt.Fprintf(cmdOut, string(jsonBuffer.Bytes()))
		default:
			fmt.Fprintf(errOut, "--output for now only support json or yaml")
		}
		return nil
	})

	if err != nil {
		return err
	}

	return nil
}
