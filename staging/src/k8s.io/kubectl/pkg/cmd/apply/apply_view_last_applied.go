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
	"bytes"
	"encoding/json"
	"fmt"

	"github.com/spf13/cobra"
	"k8s.io/cli-runtime/pkg/genericiooptions"
	"k8s.io/cli-runtime/pkg/resource"
	cmdutil "k8s.io/kubectl/pkg/cmd/util"
	"k8s.io/kubectl/pkg/util"
	"k8s.io/kubectl/pkg/util/completion"
	"k8s.io/kubectl/pkg/util/i18n"
	"k8s.io/kubectl/pkg/util/templates"
	"sigs.k8s.io/yaml"
)

// ViewLastAppliedOptions defines options for the `apply view-last-applied` command.`
type ViewLastAppliedOptions struct {
	FilenameOptions              resource.FilenameOptions
	Selector                     string
	LastAppliedConfigurationList []string
	OutputFormat                 string
	All                          bool
	Factory                      cmdutil.Factory

	genericiooptions.IOStreams
}

var (
	applyViewLastAppliedLong = templates.LongDesc(i18n.T(`
		View the latest last-applied-configuration annotations by type/name or file.

		The default output will be printed to stdout in YAML format. You can use the -o option
		to change the output format.`))

	applyViewLastAppliedExample = templates.Examples(i18n.T(`
		# View the last-applied-configuration annotations by type/name in YAML
		kubectl apply view-last-applied deployment/nginx

		# View the last-applied-configuration annotations by file in JSON
		kubectl apply view-last-applied -f deploy.yaml -o json`))
)

// NewViewLastAppliedOptions takes option arguments from a CLI stream and returns it at ViewLastAppliedOptions type.
func NewViewLastAppliedOptions(ioStreams genericiooptions.IOStreams) *ViewLastAppliedOptions {
	return &ViewLastAppliedOptions{
		OutputFormat: "yaml",

		IOStreams: ioStreams,
	}
}

// NewCmdApplyViewLastApplied creates the cobra CLI `apply` subcommand `view-last-applied`.`
func NewCmdApplyViewLastApplied(f cmdutil.Factory, ioStreams genericiooptions.IOStreams) *cobra.Command {
	options := NewViewLastAppliedOptions(ioStreams)

	cmd := &cobra.Command{
		Use:                   "view-last-applied (TYPE [NAME | -l label] | TYPE/NAME | -f FILENAME)",
		DisableFlagsInUseLine: true,
		Short:                 i18n.T("View the latest last-applied-configuration annotations of a resource/object"),
		Long:                  applyViewLastAppliedLong,
		Example:               applyViewLastAppliedExample,
		ValidArgsFunction:     completion.ResourceTypeAndNameCompletionFunc(f),
		Run: func(cmd *cobra.Command, args []string) {
			cmdutil.CheckErr(options.Complete(cmd, f, args))
			cmdutil.CheckErr(options.Validate())
			cmdutil.CheckErr(options.RunApplyViewLastApplied(cmd))
		},
	}

	cmd.Flags().StringVarP(&options.OutputFormat, "output", "o", options.OutputFormat, `Output format. Must be one of (yaml, json)`)
	cmd.Flags().BoolVar(&options.All, "all", options.All, "Select all resources in the namespace of the specified resource types")
	usage := "that contains the last-applied-configuration annotations"
	cmdutil.AddFilenameOptionFlags(cmd, &options.FilenameOptions, usage)
	cmdutil.AddLabelSelectorFlagVar(cmd, &options.Selector)

	return cmd
}

// Complete checks an object for last-applied-configuration annotations.
func (o *ViewLastAppliedOptions) Complete(cmd *cobra.Command, f cmdutil.Factory, args []string) error {
	cmdNamespace, enforceNamespace, err := f.ToRawKubeConfigLoader().Namespace()
	if err != nil {
		return err
	}

	r := f.NewBuilder().
		Unstructured().
		NamespaceParam(cmdNamespace).DefaultNamespace().
		FilenameParam(enforceNamespace, &o.FilenameOptions).
		ResourceTypeOrNameArgs(enforceNamespace, args...).
		SelectAllParam(o.All).
		LabelSelectorParam(o.Selector).
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

		configString, err := util.GetOriginalConfiguration(info.Object)
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

// Validate checks ViewLastAppliedOptions for validity.
func (o *ViewLastAppliedOptions) Validate() error {
	return nil
}

// RunApplyViewLastApplied executes the `view-last-applied` command according to ViewLastAppliedOptions.
func (o *ViewLastAppliedOptions) RunApplyViewLastApplied(cmd *cobra.Command) error {
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
		default:
			return cmdutil.UsageErrorf(
				cmd,
				"Unexpected -o output mode: %s, the flag 'output' must be one of yaml|json",
				o.OutputFormat)
		}
	}

	return nil
}
