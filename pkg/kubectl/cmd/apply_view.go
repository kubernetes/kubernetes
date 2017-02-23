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
	"fmt"

	"github.com/spf13/cobra"
	"k8s.io/kubernetes/pkg/kubectl/cmd/templates"
	"k8s.io/kubernetes/pkg/kubectl/resource"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/kubernetes/pkg/api/annotations"
	"k8s.io/kubernetes/pkg/kubectl"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
)

var (
	applyViewLong = templates.LongDesc(`
		Get the last-applied-configuration annotations by type/name or file.

		You can use --output jsonpath={...} to extract specific values using a jsonpath expression.`)

	applyViewExample = templates.Examples(`
		# Get the last-applied-configuration annotations by type/name
		kubectl apply view last-applied deployment/nginx

		# Get the last-applied-configuration annotations by file
		kubectl apply view last-applied -f deployment_nginx.yaml -o json`)

	lastApplied = "last-applied"
)

func NewCmdApplyView(f cmdutil.Factory, cmdOut io.Writer) *cobra.Command {
	options := &ApplyOptions {}
	cmd := &cobra.Command{
		Use:     "view (TYPE [NAME | -l label] | TYPE/NAME | -f FILENAME)",
		Short:   "Get the last-applied-configuration annotations",
		Long:    applyViewLong,
		Example: applyViewExample,
		Run: func(cmd *cobra.Command, args []string) {
			err := ApplyViewLastApplied(f, cmdOut, cmd, args, options)
			cmdutil.CheckErr(err)
		},
	}
	cmdutil.AddPrinterFlags(cmd)
	usage := "that contains the last-applied-configuration annotations"
	cmdutil.AddFilenameOptionFlags(cmd, &options.FilenameOptions, usage)
	return cmd
}

func ApplyViewLastApplied(f cmdutil.Factory, errOut io.Writer, cmd *cobra.Command, args []string, options *ApplyOptions) error {
	if args[0] != lastApplied {
		return cmdutil.UsageError(cmd, "for now apply %s only support last-applied")
	}

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
		ResourceTypeOrNameArgs(true, args[1:]...).
		SingleResourceType().
		Latest().
		Do()
	err = r.Err()
	if err != nil {
		return err
	}
	infos, err := r.Infos()
	if err != nil {
		return err
	}
	info := infos[0]
	encoder := f.JSONEncoder()
	_, err = kubectl.GetModifiedConfiguration(info, false, encoder)
	if err != nil {
		return err
	}

	annotationMap, err := info.Mapping.MetadataAccessor.Annotations(info.Object)
	if err != nil {
		return err
	}

	if _, ok := annotationMap[annotations.LastAppliedConfigAnnotation]; !ok {
		fmt.Fprintf(errOut, warningNoLastAppliedConfigAnnotation)
	}

	/*
	outputFormat := cmdutil.GetFlagString(cmd, "output")
	o.printer, _, err = kubectl.GetPrinter(outputFormat, templateFile, false, cmdutil.GetFlagBool(cmd, "allow-missing-template-keys"))
	if err != nil {
		return err
	}
	printer, generic, err := cmdutil.PrinterForCommand(cmd)
	if err != nil {
		return err
	}*/

	fmt.Println(annotationMap[annotations.LastAppliedConfigAnnotation])

	return nil
}