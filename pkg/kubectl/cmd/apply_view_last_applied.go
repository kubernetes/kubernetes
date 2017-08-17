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
	"k8s.io/kubernetes/pkg/kubectl"
	"k8s.io/kubernetes/pkg/kubectl/cmd/templates"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/kubectl/resource"
	"k8s.io/kubernetes/pkg/kubectl/util/i18n"
)

type ViewLastAppliedOptions struct {
	FilenameOptions              resource.FilenameOptions
	Selector                     string
	LastAppliedConfigurationList []string
	OutputFormat                 string
	All                          bool
	Factory                      cmdutil.Factory
	Out                          io.Writer
	ErrOut                       io.Writer
}

var (
	applyViewLastAppliedLong = templates.LongDesc(i18n.T(`
		View the latest last-applied-configuration annotations by type/name or file.

		The default output will be printed to stdout in YAML format. One can use -o option
		to change output format.`))

	applyViewLastAppliedExample = templates.Examples(i18n.T(`
		# View the last-applied-configuration annotations by type/name in YAML.
		kubectl apply view-last-applied deployment/nginx

		# View the last-applied-configuration annotations by file in JSON
		kubectl apply view-last-applied -f deploy.yaml -o json`))
)

func NewCmdApplyViewLastApplied(f cmdutil.Factory, out, err io.Writer) *cobra.Command {
	options := &ViewLastAppliedOptions{Out: out, ErrOut: err}
	cmd := &cobra.Command{
		Use:     "view-last-applied (TYPE [NAME | -l label] | TYPE/NAME | -f FILENAME)",
		Short:   i18n.T("View latest last-applied-configuration annotations of a resource/object"),
		Long:    applyViewLastAppliedLong,
		Example: applyViewLastAppliedExample,
		Run: func(cmd *cobra.Command, args []string) {
			cmdutil.CheckErr(options.ValidateOutputArgs(cmd))
			cmdutil.CheckErr(options.Complete(f, args))
			cmdutil.CheckErr(options.Validate(cmd))
			cmdutil.CheckErr(options.RunApplyViewLastApplied())
		},
	}

	cmd.Flags().StringP("output", "o", "", "Output format. Must be one of yaml|json")
	cmd.Flags().StringVarP(&options.Selector, "selector", "l", "", "Selector (label query) to filter on, supports '=', '==', and '!='.(e.g. -l key1=value1,key2=value2)")
	cmd.Flags().BoolVar(&options.All, "all", false, "Select all resources in the namespace of the specified resource types")
	usage := "that contains the last-applied-configuration annotations"
	cmdutil.AddFilenameOptionFlags(cmd, &options.FilenameOptions, usage)

	return cmd
}

func (o *ViewLastAppliedOptions) Complete(f cmdutil.Factory, args []string) error {
	cmdNamespace, enforceNamespace, err := f.DefaultNamespace()
	if err != nil {
		return err
	}

	builder, err := f.NewUnstructuredBuilder(true)
	if err != nil {
		return err
	}

	r := builder.
		NamespaceParam(cmdNamespace).DefaultNamespace().
		FilenameParam(enforceNamespace, &o.FilenameOptions).
		ResourceTypeOrNameArgs(enforceNamespace, args...).
		SelectAllParam(o.All).
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
			return cmdutil.AddSourceToErr(fmt.Sprintf("no last-applied-configuration annotation found on resource: %s\n", info.Name), info.Source, err)
		}
		o.LastAppliedConfigurationList = append(o.LastAppliedConfigurationList, string(configString))
		return nil
	})

	if err != nil {
		return err
	}

	return nil
}

func (o *ViewLastAppliedOptions) Validate(cmd *cobra.Command) error {
	return nil
}

func (o *ViewLastAppliedOptions) RunApplyViewLastApplied() error {
	for _, str := range o.LastAppliedConfigurationList {
		switch o.OutputFormat {
		case "json":
			jsonBuffer := &bytes.Buffer{}
			err := json.Indent(jsonBuffer, []byte(str), "", "  ")
			if err != nil {
				return err
			}
			fmt.Fprint(o.Out, string(jsonBuffer.Bytes()))
		case "yaml":
			yamlOutput, err := yaml.JSONToYAML([]byte(str))
			if err != nil {
				return err
			}
			fmt.Fprint(o.Out, string(yamlOutput))
		}
	}

	return nil
}

func (o *ViewLastAppliedOptions) ValidateOutputArgs(cmd *cobra.Command) error {
	format := cmdutil.GetFlagString(cmd, "output")
	switch format {
	case "json":
		o.OutputFormat = "json"
		return nil
	// If flag -o is not specified, use yaml as default
	case "yaml", "":
		o.OutputFormat = "yaml"
		return nil
	default:
		return cmdutil.UsageErrorf(cmd, "Unexpected -o output mode: %s, the flag 'output' must be one of yaml|json", format)
	}
}
