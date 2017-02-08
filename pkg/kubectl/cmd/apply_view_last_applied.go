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
	"k8s.io/kubernetes/pkg/kubectl"
	"k8s.io/kubernetes/pkg/kubectl/cmd/templates"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/kubectl/resource"
)

type ViewLastApplied struct {
	FilenameOptions              resource.FilenameOptions
	Selector                     string
	LastAppliedConfigurationList []string
	OutputFormat                 string
	Factory                      cmdutil.Factory
	Out                          io.Writer
	ErrOut                       io.Writer
}

var (
	applyViewLastAppliedLong = templates.LongDesc(`
		View the latest last-applied-configuration annotations by type/name or file.

		You can use --output jsonpath={...} to extract specific values using a jsonpath expression.`)

	applyViewLastAppliedExample = templates.Examples(`
		# View the latest applied configuration file for a resource by type/name.
		# This is retrieved from the last-applied-configuration annotation on the resource
		kubectl apply view-last-applied deployment/nginx

		# View the last-applied-configuration annotations by file
		kubectl apply view-last-applied -f deployment_nginx.yaml -o json`)
)

func NewCmdApplyViewLastApplied(f cmdutil.Factory, out, err io.Writer) *cobra.Command {
	options := &ViewLastApplied{Out: out, ErrOut: err}
	cmd := &cobra.Command{
		Use:     "view-last-applied (TYPE [NAME | -l label] | TYPE/NAME | -f FILENAME)",
		Short:   "View latest last-applied-configuration annotations of a resource/object",
		Long:    applyViewLastAppliedLong,
		Example: applyViewLastAppliedExample,
		Run: func(cmd *cobra.Command, args []string) {
			cmdutil.CheckErr(options.Complete(f, args))
			cmdutil.CheckErr(options.Validate(cmd))
			cmdutil.CheckErr(options.RunApplyViewLastApplied())
		},
	}

	cmdutil.AddOutputVarFlagsForMutation(cmd, &options.OutputFormat)
	cmd.Flags().StringVarP(&options.Selector, "selector", "l", "", "Selector (label query) to filter on, supports '=', '==', and '!='.")
	usage := "that contains the last-applied-configuration annotations"
	cmdutil.AddFilenameOptionFlags(cmd, &options.FilenameOptions, usage)

	return cmd
}

func (o *ViewLastApplied) Complete(f cmdutil.Factory, args []string) error {
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
		ResourceTypeOrNameArgs(enforceNamespace, args...).
		SelectorParam(o.Selector).
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

		configString, err := kubectl.GetOriginalConfiguration(info.Mapping, info.Object)
		if err != nil {
			return err
		}
		if configString == nil {
			return fmt.Errorf("no last-applied-configuration annotation found on resource: %s", info.Name)
		}
		o.LastAppliedConfigurationList = append(o.LastAppliedConfigurationList, string(configString))
		return nil
	})

	if err != nil {
		return err
	}

	return nil
}

func (o *ViewLastApplied) Validate(cmd *cobra.Command) error {
	o.OutputFormat = cmdutil.GetFlagString(cmd, "output")
	if o.OutputFormat == "" {
		o.OutputFormat = "yaml"
	}
	if !(o.OutputFormat == "yaml" || o.OutputFormat == "json") {
		return fmt.Errorf("--output for now only support json or yaml\n")
	}
	return nil
}

func (o *ViewLastApplied) RunApplyViewLastApplied() error {
	for _, str := range o.LastAppliedConfigurationList {
		yamlOutput, err := yaml.JSONToYAML([]byte(str))
		switch o.OutputFormat {
		case "json":
			jsonBuffer := &bytes.Buffer{}
			err = json.Indent(jsonBuffer, []byte(str), "", "  ")
			if err != nil {
				return err
			}
			fmt.Fprintf(o.Out, string(jsonBuffer.Bytes()))
		case "yaml":
			fmt.Println(string(yamlOutput))
			fmt.Fprintf(o.Out, string(yamlOutput))
		}
	}

	return nil
}
