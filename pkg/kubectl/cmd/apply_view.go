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
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/kubectl/resource"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/kubernetes/pkg/api/annotations"
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
		Use:     "view",
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

	allNamespaces := cmdutil.GetFlagBool(cmd, "all-namespaces")
	mapper, typer, err := f.UnstructuredObject()
	if err != nil {
		return err
	}

	cmdNamespace, enforceNamespace, err := f.DefaultNamespace()
	if err != nil {
		return err
	}

	if allNamespaces {
		enforceNamespace = false
	}

	r := resource.NewBuilder(mapper, typer, resource.ClientMapperFunc(f.UnstructuredClientForMapping), unstructured.UnstructuredJSONScheme).
		NamespaceParam(cmdNamespace).DefaultNamespace().AllNamespaces(allNamespaces).
		FilenameParam(enforceNamespace, &options.FilenameOptions).
		ResourceTypeOrNameArgs(true, args...).
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

	if len(infos) != 1 {
		//return i18n.Errorf("watch is only supported on individual resources and resource collections - %d resources were found", len(infos))
	}
	info := infos[0]

	annotationMap, err := info.Mapping.MetadataAccessor.Annotations(info.Object)
	if err != nil {
		return err
	}
	if _, ok := annotationMap[annotations.LastAppliedConfigAnnotation]; !ok {
		fmt.Fprintf(errOut, warningNoLastAppliedConfigAnnotation)
	}

	fmt.Println(annotationMap[annotations.LastAppliedConfigAnnotation])

	/*
	mapping := info.ResourceMapping()
	printer, err := f.PrinterForMapping(cmd, mapping, allNamespaces)
	if err != nil {
		return err
	}
	obj, err := r.Object()
	if err != nil {
		return err
	}
	*/

	return nil
}