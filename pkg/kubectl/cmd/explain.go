/*
Copyright 2014 The Kubernetes Authors.

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

	"github.com/spf13/cobra"

	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/kubectl"
	"k8s.io/kubernetes/pkg/kubectl/cmd/templates"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/util/i18n"
)

var (
	explainLong = templates.LongDesc(`
		Documentation of resources.

		` + valid_resources)

	explainExamples = templates.Examples(`
		# Get the documentation of the resource and its fields
		kubectl explain pods

		# Get the documentation of a specific field of a resource
		kubectl explain pods.spec.containers`)
)

// NewCmdExplain returns a cobra command for swagger docs
func NewCmdExplain(f cmdutil.Factory, out, cmdErr io.Writer) *cobra.Command {
	cmd := &cobra.Command{
		Use:     "explain RESOURCE",
		Short:   i18n.T("Documentation of resources"),
		Long:    explainLong,
		Example: explainExamples,
		Run: func(cmd *cobra.Command, args []string) {
			err := RunExplain(f, out, cmdErr, cmd, args)
			cmdutil.CheckErr(err)
		},
	}
	cmd.Flags().Bool("recursive", false, "Print the fields of fields (Currently only 1 level deep)")
	cmdutil.AddInclude3rdPartyFlags(cmd)
	return cmd
}

// RunExplain executes the appropriate steps to print a model's documentation
func RunExplain(f cmdutil.Factory, out, cmdErr io.Writer, cmd *cobra.Command, args []string) error {
	if len(args) == 0 {
		fmt.Fprint(cmdErr, "You must specify the type of resource to explain. ", valid_resources)
		return cmdutil.UsageError(cmd, "Required resource not specified.")
	}
	if len(args) > 1 {
		return cmdutil.UsageError(cmd, "We accept only this format: explain RESOURCE")
	}

	recursive := cmdutil.GetFlagBool(cmd, "recursive")
	apiVersionString := cmdutil.GetFlagString(cmd, "api-version")
	apiVersion := schema.GroupVersion{}

	mapper, _ := f.Object()
	// TODO: After we figured out the new syntax to separate group and resource, allow
	// the users to use it in explain (kubectl explain <group><syntax><resource>).
	// Refer to issue #16039 for why we do this. Refer to PR #15808 that used "/" syntax.
	inModel, fieldsPath, err := kubectl.SplitAndParseResourceRequest(args[0], mapper)
	if err != nil {
		return err
	}

	// TODO: We should deduce the group for a resource by discovering the supported resources at server.
	fullySpecifiedGVR, groupResource := schema.ParseResourceArg(inModel)
	gvk := schema.GroupVersionKind{}
	if fullySpecifiedGVR != nil {
		gvk, _ = mapper.KindFor(*fullySpecifiedGVR)
	}
	if gvk.Empty() {
		gvk, err = mapper.KindFor(groupResource.WithVersion(""))
		if err != nil {
			return err
		}
	}

	if len(apiVersionString) == 0 {
		groupMeta, err := api.Registry.Group(gvk.Group)
		if err != nil {
			return err
		}
		apiVersion = groupMeta.GroupVersion

	} else {
		apiVersion, err = schema.ParseGroupVersion(apiVersionString)
		if err != nil {
			return nil
		}
	}

	schema, err := f.SwaggerSchema(apiVersion.WithKind(gvk.Kind))
	if err != nil {
		return err
	}

	return kubectl.PrintModelDescription(inModel, fieldsPath, out, schema, recursive)
}
