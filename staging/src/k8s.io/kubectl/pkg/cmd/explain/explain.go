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

package explain

import (
	"fmt"

	"github.com/spf13/cobra"

	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/cli-runtime/pkg/genericclioptions"
	cmdutil "k8s.io/kubectl/pkg/cmd/util"
	"k8s.io/kubectl/pkg/explain"
	"k8s.io/kubectl/pkg/util/i18n"
	"k8s.io/kubectl/pkg/util/openapi"
	"k8s.io/kubectl/pkg/util/templates"
)

var (
	explainLong = templates.LongDesc(i18n.T(`
		List the fields for supported resources

		This command describes the fields associated with each supported API resource.
		Fields are identified via a simple JSONPath identifier:

			<type>.<fieldName>[.<fieldName>]

		Add the --recursive flag to display all of the fields at once without descriptions.
		Information about each field is retrieved from the server in OpenAPI format.`))

	explainExamples = templates.Examples(i18n.T(`
		# Get the documentation of the resource and its fields
		kubectl explain pods

		# Get the documentation of a specific field of a resource
		kubectl explain pods.spec.containers`))
)

type ExplainOptions struct {
	genericclioptions.IOStreams

	CmdParent  string
	APIVersion string
	Recursive  bool

	Mapper meta.RESTMapper
	Schema openapi.Resources
}

func NewExplainOptions(parent string, streams genericclioptions.IOStreams) *ExplainOptions {
	return &ExplainOptions{
		IOStreams: streams,
		CmdParent: parent,
	}
}

// NewCmdExplain returns a cobra command for swagger docs
func NewCmdExplain(parent string, f cmdutil.Factory, streams genericclioptions.IOStreams) *cobra.Command {
	o := NewExplainOptions(parent, streams)

	cmd := &cobra.Command{
		Use:                   "explain RESOURCE",
		DisableFlagsInUseLine: true,
		Short:                 i18n.T("Documentation of resources"),
		Long:                  explainLong + "\n\n" + cmdutil.SuggestAPIResources(parent),
		Example:               explainExamples,
		Run: func(cmd *cobra.Command, args []string) {
			cmdutil.CheckErr(o.Complete(f, cmd))
			cmdutil.CheckErr(o.Validate(args))
			cmdutil.CheckErr(o.Run(args))
		},
	}
	cmd.Flags().BoolVar(&o.Recursive, "recursive", o.Recursive, "Print the fields of fields (Currently only 1 level deep)")
	cmd.Flags().StringVar(&o.APIVersion, "api-version", o.APIVersion, "Get different explanations for particular API version (API group/version)")
	return cmd
}

func (o *ExplainOptions) Complete(f cmdutil.Factory, cmd *cobra.Command) error {
	var err error
	o.Mapper, err = f.ToRESTMapper()
	if err != nil {
		return err
	}

	o.Schema, err = f.OpenAPISchema()
	if err != nil {
		return err
	}
	return nil
}

func (o *ExplainOptions) Validate(args []string) error {
	if len(args) == 0 {
		return fmt.Errorf("You must specify the type of resource to explain. %s\n", cmdutil.SuggestAPIResources(o.CmdParent))
	}
	if len(args) > 1 {
		return fmt.Errorf("We accept only this format: explain RESOURCE\n")
	}

	return nil
}

// Run executes the appropriate steps to print a model's documentation
func (o *ExplainOptions) Run(args []string) error {
	recursive := o.Recursive
	apiVersionString := o.APIVersion

	// TODO: After we figured out the new syntax to separate group and resource, allow
	// the users to use it in explain (kubectl explain <group><syntax><resource>).
	// Refer to issue #16039 for why we do this. Refer to PR #15808 that used "/" syntax.
	fullySpecifiedGVR, fieldsPath, err := explain.SplitAndParseResourceRequest(args[0], o.Mapper)
	if err != nil {
		return err
	}

	gvk, _ := o.Mapper.KindFor(fullySpecifiedGVR)
	if gvk.Empty() {
		gvk, err = o.Mapper.KindFor(fullySpecifiedGVR.GroupResource().WithVersion(""))
		if err != nil {
			return err
		}
	}

	if len(apiVersionString) != 0 {
		apiVersion, err := schema.ParseGroupVersion(apiVersionString)
		if err != nil {
			return err
		}
		gvk = apiVersion.WithKind(gvk.Kind)
	}

	schema := o.Schema.LookupResource(gvk)
	if schema == nil {
		return fmt.Errorf("couldn't find resource for %q", gvk)
	}

	return explain.PrintModelDescription(fieldsPath, o.Out, schema, gvk, recursive)
}
