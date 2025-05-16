/*
Copyright 2018 The Kubernetes Authors.

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

package apiresources

import (
	"fmt"
	"io"
	"sort"
	"strings"

	"github.com/spf13/cobra"
	"k8s.io/apimachinery/pkg/runtime"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/cli-runtime/pkg/genericclioptions"
	"k8s.io/cli-runtime/pkg/genericiooptions"
	"k8s.io/cli-runtime/pkg/printers"
	"k8s.io/client-go/discovery"
	cmdutil "k8s.io/kubectl/pkg/cmd/util"
	"k8s.io/kubectl/pkg/util/i18n"
	"k8s.io/kubectl/pkg/util/templates"
)

var (
	apiresourcesExample = templates.Examples(`
		# Print the supported API resources
		kubectl api-resources

		# Print the supported API resources with more information
		kubectl api-resources -o wide

		# Print the supported API resources sorted by a column
		kubectl api-resources --sort-by=name

		# Print the supported namespaced resources
		kubectl api-resources --namespaced=true

		# Print the supported non-namespaced resources
		kubectl api-resources --namespaced=false

		# Print the supported API resources with a specific APIGroup
		kubectl api-resources --api-group=rbac.authorization.k8s.io`)
)

// APIResourceOptions is the start of the data required to perform the operation.
// As new fields are added, add them here instead of referencing the cmd.Flags()
type APIResourceOptions struct {
	SortBy     string
	APIGroup   string
	Namespaced bool
	Verbs      []string
	Cached     bool
	Categories []string

	groupChanged bool
	nsChanged    bool

	discoveryClient discovery.CachedDiscoveryInterface

	genericiooptions.IOStreams
	PrintFlags *PrintFlags
	PrintObj   printers.ResourcePrinterFunc
}

type PrintFlags struct {
	JSONYamlPrintFlags *genericclioptions.JSONYamlPrintFlags
	NamePrintFlags     NamePrintFlags
	HumanReadableFlags HumanPrintFlags

	NoHeaders    *bool
	OutputFormat *string
}

func NewPrintFlags() *PrintFlags {
	outputFormat := ""
	noHeaders := false

	return &PrintFlags{
		OutputFormat:       &outputFormat,
		NoHeaders:          &noHeaders,
		JSONYamlPrintFlags: genericclioptions.NewJSONYamlPrintFlags(),
		NamePrintFlags:     APIResourcesNewNamePrintFlags(),
		HumanReadableFlags: APIResourcesHumanReadableFlags(),
	}
}

// PrintOptions struct defines a struct for various print options
type PrintOptions struct {
	SortBy    *string
	NoHeaders bool
	Wide      bool
}

type HumanPrintFlags struct {
	SortBy    *string
	NoHeaders bool
}

func (f *HumanPrintFlags) AllowedFormats() []string {
	return []string{"wide"}
}

// AddFlags receives a *cobra.Command reference and binds
// flags related to human-readable printing to it
func (f *HumanPrintFlags) AddFlags(c *cobra.Command) {
	if f.SortBy != nil {
		c.Flags().StringVar(f.SortBy, "sort-by", *f.SortBy, "If non-empty, sort list types using this field specification.  The field specification is expressed as a JSONPath expression (e.g. '{.metadata.name}'). The field in the API resource specified by this JSONPath expression must be an integer or a string.")
	}
}

// ToPrinter receives an outputFormat and returns a printer capable of
// handling human-readable output.
func (f *HumanPrintFlags) ToPrinter(outputFormat string) (printers.ResourcePrinter, error) {
	if len(outputFormat) > 0 && outputFormat != "wide" {
		return nil, genericclioptions.NoCompatiblePrinterError{Options: f, AllowedFormats: f.AllowedFormats()}
	}

	p := HumanReadablePrinter{
		options: PrintOptions{
			NoHeaders: f.NoHeaders,
			Wide:      outputFormat == "wide",
		},
	}

	return p, nil
}

type HumanReadablePrinter struct {
	options        PrintOptions
	lastType       interface{}
	lastColumns    []metav1.TableColumnDefinition
	printedHeaders bool
}

func (f HumanReadablePrinter) PrintObj(obj runtime.Object, w io.Writer) error {
	flatList, ok := obj.(*metav1.APIResourceList)
	if !ok {
		return fmt.Errorf("object is not a APIResourceList")
	}
	var errs []error
	for _, r := range flatList.APIResources {
		gv, err := schema.ParseGroupVersion(strings.Join([]string{r.Group, r.Version}, "/"))
		if err != nil {
			errs = append(errs, err)
			continue
		}
		if f.options.Wide {
			if _, err := fmt.Fprintf(w, "%s\t%s\t%s\t%v\t%s\t%v\t%v\n",
				r.Name,
				strings.Join(r.ShortNames, ","),
				gv.String(),
				r.Namespaced,
				r.Kind,
				strings.Join(r.Verbs, ","),
				strings.Join(r.Categories, ",")); err != nil {
				errs = append(errs, err)
			}
			continue
		}
		if _, err := fmt.Fprintf(w, "%s\t%s\t%s\t%v\t%s\n",
			r.Name,
			strings.Join(r.ShortNames, ","),
			gv.String(),
			r.Namespaced,
			r.Kind); err != nil {
			errs = append(errs, err)
		}
	}
	return utilerrors.NewAggregate(errs)
}

type NamePrintFlags struct{}

func APIResourcesNewNamePrintFlags() NamePrintFlags {
	return NamePrintFlags{}
}

func (f *NamePrintFlags) AllowedFormats() []string {
	return []string{"name"}
}

// AddFlags receives a *cobra.Command reference and binds
// flags related to name printing to it
func (f *NamePrintFlags) AddFlags(_ *cobra.Command) {}

// ToPrinter receives an outputFormat and returns a printer capable of
// handling human-readable output.
func (f *NamePrintFlags) ToPrinter(outputFormat string) (printers.ResourcePrinter, error) {
	if outputFormat == "name" {
		return NamePrinter{}, nil
	}
	return nil, genericclioptions.NoCompatiblePrinterError{Options: f, AllowedFormats: f.AllowedFormats()}
}

type NamePrinter struct{}

func (f NamePrinter) PrintObj(obj runtime.Object, w io.Writer) error {
	flatList, ok := obj.(*metav1.APIResourceList)
	if !ok {
		return fmt.Errorf("object is not a APIResourceList")
	}
	var errs []error
	for _, r := range flatList.APIResources {
		name := r.Name
		if len(r.Group) > 0 {
			name += "." + r.Group
		}
		if _, err := fmt.Fprintf(w, "%s\n", name); err != nil {
			errs = append(errs, err)
		}
	}
	return nil
}

func APIResourcesHumanReadableFlags() HumanPrintFlags {
	return HumanPrintFlags{
		SortBy:    nil,
		NoHeaders: false,
	}
}

func (f *PrintFlags) AllowedFormats() []string {
	ret := []string{}
	ret = append(ret, f.JSONYamlPrintFlags.AllowedFormats()...)
	ret = append(ret, f.NamePrintFlags.AllowedFormats()...)
	ret = append(ret, f.HumanReadableFlags.AllowedFormats()...)
	return ret
}

func (f *PrintFlags) AddFlags(cmd *cobra.Command) {
	f.JSONYamlPrintFlags.AddFlags(cmd)
	f.HumanReadableFlags.AddFlags(cmd)
	f.NamePrintFlags.AddFlags(cmd)

	if f.OutputFormat != nil {
		cmd.Flags().StringVarP(f.OutputFormat, "output", "o", *f.OutputFormat, fmt.Sprintf("Output format. One of: (%s).", strings.Join(f.AllowedFormats(), ", ")))
	}
	if f.NoHeaders != nil {
		cmd.Flags().BoolVar(f.NoHeaders, "no-headers", *f.NoHeaders, "When using the default or custom-column output format, don't print headers (default print headers).")
	}
}

func (f *PrintFlags) ToPrinter() (printers.ResourcePrinter, error) {
	outputFormat := ""
	if f.OutputFormat != nil {
		outputFormat = *f.OutputFormat
	}

	noHeaders := false
	if f.NoHeaders != nil {
		noHeaders = *f.NoHeaders
	}
	f.HumanReadableFlags.NoHeaders = noHeaders

	if p, err := f.JSONYamlPrintFlags.ToPrinter(outputFormat); !genericclioptions.IsNoCompatiblePrinterError(err) {
		return p, err
	}

	if p, err := f.HumanReadableFlags.ToPrinter(outputFormat); !genericclioptions.IsNoCompatiblePrinterError(err) {
		return p, err
	}

	if p, err := f.NamePrintFlags.ToPrinter(outputFormat); !genericclioptions.IsNoCompatiblePrinterError(err) {
		return p, err
	}

	return nil, genericclioptions.NoCompatiblePrinterError{OutputFormat: &outputFormat, AllowedFormats: f.AllowedFormats()}
}

// NewAPIResourceOptions creates the options for APIResource
func NewAPIResourceOptions(ioStreams genericiooptions.IOStreams) *APIResourceOptions {
	return &APIResourceOptions{
		IOStreams:  ioStreams,
		Namespaced: true,
		PrintFlags: NewPrintFlags(),
	}
}

// NewCmdAPIResources creates the `api-resources` command
func NewCmdAPIResources(restClientGetter genericclioptions.RESTClientGetter, ioStreams genericiooptions.IOStreams) *cobra.Command {
	o := NewAPIResourceOptions(ioStreams)

	cmd := &cobra.Command{
		Use:     "api-resources",
		Short:   i18n.T("Print the supported API resources on the server"),
		Long:    i18n.T("Print the supported API resources on the server."),
		Example: apiresourcesExample,
		Run: func(cmd *cobra.Command, args []string) {
			cmdutil.CheckErr(o.Complete(restClientGetter, cmd, args))
			cmdutil.CheckErr(o.Validate())
			cmdutil.CheckErr(o.RunAPIResources())
		},
	}

	o.PrintFlags.AddFlags(cmd)

	cmd.Flags().StringVar(&o.APIGroup, "api-group", o.APIGroup, "Limit to resources in the specified API group.")
	cmd.Flags().BoolVar(&o.Namespaced, "namespaced", o.Namespaced, "If false, non-namespaced resources will be returned, otherwise returning namespaced resources by default.")
	cmd.Flags().StringSliceVar(&o.Verbs, "verbs", o.Verbs, "Limit to resources that support the specified verbs.")
	cmd.Flags().StringVar(&o.SortBy, "sort-by", o.SortBy, "If non-empty, sort list of resources using specified field. The field can be either 'name' or 'kind'.")
	cmd.Flags().BoolVar(&o.Cached, "cached", o.Cached, "Use the cached list of resources if available.")
	cmd.Flags().StringSliceVar(&o.Categories, "categories", o.Categories, "Limit to resources that belong to the specified categories.")
	return cmd
}

// Validate checks to the APIResourceOptions to see if there is sufficient information run the command
func (o *APIResourceOptions) Validate() error {
	supportedSortTypes := sets.New[string]("", "name", "kind")
	if len(o.SortBy) > 0 {
		if !supportedSortTypes.Has(o.SortBy) {
			return fmt.Errorf("--sort-by accepts only name or kind")
		}
	}
	return nil
}

// Complete adapts from the command line args and validates them
func (o *APIResourceOptions) Complete(restClientGetter genericclioptions.RESTClientGetter, cmd *cobra.Command, args []string) error {
	if len(args) != 0 {
		return cmdutil.UsageErrorf(cmd, "unexpected arguments: %v", args)
	}

	discoveryClient, err := restClientGetter.ToDiscoveryClient()
	if err != nil {
		return err
	}
	o.discoveryClient = discoveryClient

	o.groupChanged = cmd.Flags().Changed("api-group")
	o.nsChanged = cmd.Flags().Changed("namespaced")

	var printer printers.ResourcePrinter
	if o.PrintFlags.OutputFormat != nil {
		printer, err = o.PrintFlags.ToPrinter()
		if err != nil {
			return err
		}

		o.PrintObj = func(object runtime.Object, out io.Writer) error {
			return printer.PrintObj(object, out)
		}
	}

	return nil
}

// RunAPIResources does the work
func (o *APIResourceOptions) RunAPIResources() error {
	w := printers.GetNewTabWriter(o.Out)
	defer w.Flush()

	if !o.Cached {
		// Always request fresh data from the server
		o.discoveryClient.Invalidate()
	}

	errs := []error{}
	lists, err := o.discoveryClient.ServerPreferredResources()
	if err != nil {
		errs = append(errs, err)
	}

	var allResources []*metav1.APIResourceList

	for _, list := range lists {
		if len(list.APIResources) == 0 {
			continue
		}
		gv, err := schema.ParseGroupVersion(list.GroupVersion)
		if err != nil {
			continue
		}
		apiList := &metav1.APIResourceList{
			TypeMeta: metav1.TypeMeta{
				Kind:       "APIResourceList",
				APIVersion: "v1",
			},
			GroupVersion: gv.String(),
		}
		var apiResources []metav1.APIResource
		for _, resource := range list.APIResources {
			if len(resource.Verbs) == 0 {
				continue
			}
			// filter apiGroup
			if o.groupChanged && o.APIGroup != gv.Group {
				continue
			}
			// filter namespaced
			if o.nsChanged && o.Namespaced != resource.Namespaced {
				continue
			}
			// filter to resources that support the specified verbs
			if len(o.Verbs) > 0 && !sets.New[string](resource.Verbs...).HasAll(o.Verbs...) {
				continue
			}
			// filter to resources that belong to the specified categories
			if len(o.Categories) > 0 && !sets.New[string](resource.Categories...).HasAll(o.Categories...) {
				continue
			}
			// set these because we display a concatenation of these two values under APIVERSION column of human-readable output
			resource.Group = gv.Group
			resource.Version = gv.Version
			apiResources = append(apiResources, resource)
		}
		apiList.APIResources = apiResources
		allResources = append(allResources, apiList)
	}

	if !*o.PrintFlags.NoHeaders && (o.PrintFlags.OutputFormat == nil || *o.PrintFlags.OutputFormat == "" || *o.PrintFlags.OutputFormat == "wide") {
		if err = printContextHeaders(w, *o.PrintFlags.OutputFormat); err != nil {
			return err
		}
	}

	flatList := &metav1.APIResourceList{
		TypeMeta: metav1.TypeMeta{
			APIVersion: allResources[0].APIVersion,
			Kind:       allResources[0].Kind,
		},
	}
	for _, resource := range allResources {
		flatList.APIResources = append(flatList.APIResources, resource.APIResources...)
	}

	sort.Stable(sortableResource{flatList.APIResources, o.SortBy})

	return o.PrintObj(flatList, w)
}

func printContextHeaders(out io.Writer, output string) error {
	columnNames := []string{"NAME", "SHORTNAMES", "APIVERSION", "NAMESPACED", "KIND"}
	if output == "wide" {
		columnNames = append(columnNames, "VERBS", "CATEGORIES")
	}
	_, err := fmt.Fprintf(out, "%s\n", strings.Join(columnNames, "\t"))
	return err
}

type sortableResource struct {
	resources []metav1.APIResource
	sortBy    string
}

func (s sortableResource) Len() int { return len(s.resources) }
func (s sortableResource) Swap(i, j int) {
	s.resources[i], s.resources[j] = s.resources[j], s.resources[i]
}
func (s sortableResource) Less(i, j int) bool {
	ret := strings.Compare(s.compareValues(i, j))
	if ret > 0 {
		return false
	} else if ret == 0 {
		return strings.Compare(s.resources[i].Name, s.resources[j].Name) < 0
	}
	return true
}

func (s sortableResource) compareValues(i, j int) (string, string) {
	switch s.sortBy {
	case "name":
		return s.resources[i].Name, s.resources[j].Name
	case "kind":
		return s.resources[i].Kind, s.resources[j].Kind
	}
	return s.resources[i].Group, s.resources[j].Group
}
