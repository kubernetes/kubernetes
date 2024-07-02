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
	"github.com/spf13/pflag"

	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/cli-runtime/pkg/genericiooptions"
	openapiclient "k8s.io/client-go/openapi"
	cmdutil "k8s.io/kubectl/pkg/cmd/util"
	"k8s.io/kubectl/pkg/explain"
	openapiv3explain "k8s.io/kubectl/pkg/explain/v2"
	"k8s.io/kubectl/pkg/util/i18n"
	"k8s.io/kubectl/pkg/util/openapi"
	"k8s.io/kubectl/pkg/util/templates"
)

var (
	explainLong = templates.LongDesc(i18n.T(`
		Describe fields and structure of various resources.

		This command describes the fields associated with each supported API resource.
		Fields are identified via a simple JSONPath identifier:

			<type>.<fieldName>[.<fieldName>]

		Information about each field is retrieved from the server in OpenAPI format.`))

	explainExamples = templates.Examples(i18n.T(`
		# Get the documentation of the resource and its fields
		kubectl explain pods

		# Get all the fields in the resource
		kubectl explain pods --recursive

		# Get the explanation for deployment in supported api versions
		kubectl explain deployments --api-version=apps/v1

		# Get the documentation of a specific field of a resource
		kubectl explain pods.spec.containers
		
		# Get the documentation of resources in different format
		kubectl explain deployment --output=plaintext-openapiv2`))

	plaintextTemplateName          = "plaintext"
	plaintextOpenAPIV2TemplateName = "plaintext-openapiv2"
)

type ExplainOptions struct {
	genericiooptions.IOStreams

	CmdParent  string
	APIVersion string
	Recursive  bool

	// Flags hold the parsed CLI flags.
	Flags *pflag.FlagSet

	args []string

	Mapper        meta.RESTMapper
	openAPIGetter openapi.OpenAPIResourcesGetter

	// Name of the template to use with the openapiv3 template renderer.
	OutputFormat string

	// Client capable of fetching openapi documents from the user's cluster
	OpenAPIV3Client openapiclient.Client
}

func NewExplainOptions(parent string, streams genericiooptions.IOStreams) *ExplainOptions {
	return &ExplainOptions{
		IOStreams:    streams,
		CmdParent:    parent,
		OutputFormat: plaintextTemplateName,
	}
}

// NewCmdExplain returns a cobra command for swagger docs
func NewCmdExplain(parent string, f cmdutil.Factory, streams genericiooptions.IOStreams) *cobra.Command {
	o := NewExplainOptions(parent, streams)

	cmd := &cobra.Command{
		Use:                   "explain TYPE [--recursive=FALSE|TRUE] [--api-version=api-version-group] [--output=plaintext|plaintext-openapiv2]",
		DisableFlagsInUseLine: true,
		Short:                 i18n.T("Get documentation for a resource"),
		Long:                  explainLong + "\n\n" + cmdutil.SuggestAPIResources(parent),
		Example:               explainExamples,
		Run: func(cmd *cobra.Command, args []string) {
			cmdutil.CheckErr(o.Complete(f, cmd, args))
			cmdutil.CheckErr(o.Validate())
			cmdutil.CheckErr(o.Run())
		},
	}
	cmd.Flags().BoolVar(&o.Recursive, "recursive", o.Recursive, "When true, print the name of all the fields recursively. When false print all fields with their desciprion recursively. Otherwise, print the available fields with their description.")
	cmd.Flags().StringVar(&o.APIVersion, "api-version", o.APIVersion, "Use given api-version (group/version) of the resource.")

	// Only enable --output as a valid flag if the feature is enabled
	cmd.Flags().StringVar(&o.OutputFormat, "output", plaintextTemplateName, "Format in which to render the schema. Valid values are: (plaintext, plaintext-openapiv2).")

	// To check if field is changed by cmd.
	o.Flags = cmd.Flags()

	return cmd
}

func (o *ExplainOptions) Complete(f cmdutil.Factory, cmd *cobra.Command, args []string) error {
	var err error
	o.Mapper, err = f.ToRESTMapper()
	if err != nil {
		return err
	}

	// Only openapi v3 needs the discovery client.
	o.OpenAPIV3Client, err = f.OpenAPIV3Client()
	if err != nil {
		return err
	}

	// Lazy-load the OpenAPI V2 Resources, so they're not loaded when using OpenAPI V3.
	o.openAPIGetter = f
	o.args = args
	return nil
}

func (o *ExplainOptions) Validate() error {
	if len(o.args) == 0 {
		return fmt.Errorf("You must specify the type of resource to explain. %s\n", cmdutil.SuggestAPIResources(o.CmdParent))
	}
	if len(o.args) > 1 {
		return fmt.Errorf("We accept only this format: explain RESOURCE\n")
	}

	return nil
}

// Run executes the appropriate steps to print a model's documentation
func (o *ExplainOptions) Run() error {
	var fullySpecifiedGVR schema.GroupVersionResource
	var fieldsPath []string
	var err error
	if len(o.APIVersion) == 0 {
		fullySpecifiedGVR, fieldsPath, err = explain.SplitAndParseResourceRequestWithMatchingPrefix(o.args[0], o.Mapper)
		if err != nil {
			return err
		}
	} else {
		// TODO: After we figured out the new syntax to separate group and resource, allow
		// the users to use it in explain (kubectl explain <group><syntax><resource>).
		// Refer to issue #16039 for why we do this. Refer to PR #15808 that used "/" syntax.
		fullySpecifiedGVR, fieldsPath, err = explain.SplitAndParseResourceRequest(o.args[0], o.Mapper)
		if err != nil {
			return err
		}
	}

	// Fallback to openapiv2 implementation using special template name
	switch o.OutputFormat {
	case plaintextOpenAPIV2TemplateName:
		return o.renderOpenAPIV2(fullySpecifiedGVR, fieldsPath)
	case plaintextTemplateName:
		// Check whether the server reponds to OpenAPIV3.
		if _, err := o.OpenAPIV3Client.Paths(); err != nil {
			// Use v2 renderer if server does not support v3
			return o.renderOpenAPIV2(fullySpecifiedGVR, fieldsPath)
		}

		fallthrough
	default:
		if len(o.APIVersion) > 0 {
			apiVersion, err := schema.ParseGroupVersion(o.APIVersion)
			if err != nil {
				return err
			}
			fullySpecifiedGVR.Group = apiVersion.Group
			fullySpecifiedGVR.Version = apiVersion.Version
		}

		// Use recursive flag to decide which template to render
		if o.Flags.Changed("recursive") && !o.Recursive {
			// Use default long format template
			o.Recursive = true
		} else if o.Recursive {
			// Use short format template
			o.OutputFormat = "plaintext_short"
		}

		return openapiv3explain.PrintModelDescription(
			fieldsPath,
			o.Out,
			o.OpenAPIV3Client,
			fullySpecifiedGVR,
			o.Recursive,
			o.OutputFormat,
		)
	}
}

func (o *ExplainOptions) renderOpenAPIV2(
	fullySpecifiedGVR schema.GroupVersionResource,
	fieldsPath []string,
) error {
	var err error

	gvk, _ := o.Mapper.KindFor(fullySpecifiedGVR)
	if gvk.Empty() {
		gvk, err = o.Mapper.KindFor(fullySpecifiedGVR.GroupResource().WithVersion(""))
		if err != nil {
			return err
		}
	}

	if len(o.APIVersion) != 0 {
		apiVersion, err := schema.ParseGroupVersion(o.APIVersion)
		if err != nil {
			return err
		}
		gvk = apiVersion.WithKind(gvk.Kind)
	}

	resources, err := o.openAPIGetter.OpenAPISchema()
	if err != nil {
		return err
	}
	schema := resources.LookupResource(gvk)
	if schema == nil {
		return fmt.Errorf("couldn't find resource for %q", gvk)
	}

	return explain.PrintModelDescription(fieldsPath, o.Out, schema, gvk, o.Recursive)
}
