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

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/cli-runtime/pkg/genericclioptions"
	cmdutil "k8s.io/kubectl/pkg/cmd/util"
	"k8s.io/kubectl/pkg/util/printers"
	"k8s.io/kubectl/pkg/util/templates"
)

var (
	apiresourcesExample = templates.Examples(`
		# Print the supported API Resources
		kubectl api-resources

		# Print the supported API Resources with more information
		kubectl api-resources -o wide

		# Print the supported namespaced resources
		kubectl api-resources --namespaced=true

		# Print the supported non-namespaced resources
		kubectl api-resources --namespaced=false

		# Print the supported API Resources with specific APIGroup
		kubectl api-resources --api-group=extensions`)
)

// APIResourceOptions is the start of the data required to perform the operation.
// As new fields are added, add them here instead of referencing the cmd.Flags()
type APIResourceOptions struct {
	Output     string
	APIGroup   string
	Namespaced bool
	Verbs      []string
	NoHeaders  bool
	Cached     bool

	genericclioptions.IOStreams
}

// groupResource contains the APIGroup and APIResource
type groupResource struct {
	APIGroup    string
	APIResource metav1.APIResource
}

// NewAPIResourceOptions creates the options for APIResource
func NewAPIResourceOptions(ioStreams genericclioptions.IOStreams) *APIResourceOptions {
	return &APIResourceOptions{
		IOStreams:  ioStreams,
		Namespaced: true,
	}
}

// NewCmdAPIResources creates the `api-resources` command
func NewCmdAPIResources(f cmdutil.Factory, ioStreams genericclioptions.IOStreams) *cobra.Command {
	o := NewAPIResourceOptions(ioStreams)

	cmd := &cobra.Command{
		Use:     "api-resources",
		Short:   "Print the supported API resources on the server",
		Long:    "Print the supported API resources on the server",
		Example: apiresourcesExample,
		Run: func(cmd *cobra.Command, args []string) {
			cmdutil.CheckErr(o.Complete(cmd, args))
			cmdutil.CheckErr(o.Validate())
			cmdutil.CheckErr(o.RunAPIResources(cmd, f))
		},
	}

	cmd.Flags().BoolVar(&o.NoHeaders, "no-headers", o.NoHeaders, "When using the default or custom-column output format, don't print headers (default print headers).")
	cmd.Flags().StringVarP(&o.Output, "output", "o", o.Output, "Output format. One of: wide|name.")

	cmd.Flags().StringVar(&o.APIGroup, "api-group", o.APIGroup, "Limit to resources in the specified API group.")
	cmd.Flags().BoolVar(&o.Namespaced, "namespaced", o.Namespaced, "If false, non-namespaced resources will be returned, otherwise returning namespaced resources by default.")
	cmd.Flags().StringSliceVar(&o.Verbs, "verbs", o.Verbs, "Limit to resources that support the specified verbs.")
	cmd.Flags().BoolVar(&o.Cached, "cached", o.Cached, "Use the cached list of resources if available.")
	return cmd
}

// Validate checks to the APIResourceOptions to see if there is sufficient information run the command
func (o *APIResourceOptions) Validate() error {
	supportedOutputTypes := sets.NewString("", "wide", "name")
	if !supportedOutputTypes.Has(o.Output) {
		return fmt.Errorf("--output %v is not available", o.Output)
	}
	return nil
}

// Complete adapts from the command line args and validates them
func (o *APIResourceOptions) Complete(cmd *cobra.Command, args []string) error {
	if len(args) != 0 {
		return cmdutil.UsageErrorf(cmd, "unexpected arguments: %v", args)
	}
	return nil
}

// RunAPIResources does the work
func (o *APIResourceOptions) RunAPIResources(cmd *cobra.Command, f cmdutil.Factory) error {
	w := printers.GetNewTabWriter(o.Out)
	defer w.Flush()

	discoveryclient, err := f.ToDiscoveryClient()
	if err != nil {
		return err
	}

	if !o.Cached {
		// Always request fresh data from the server
		discoveryclient.Invalidate()
	}

	errs := []error{}
	lists, err := discoveryclient.ServerPreferredResources()
	if err != nil {
		errs = append(errs, err)
	}

	resources := []groupResource{}

	groupChanged := cmd.Flags().Changed("api-group")
	nsChanged := cmd.Flags().Changed("namespaced")

	for _, list := range lists {
		if len(list.APIResources) == 0 {
			continue
		}
		gv, err := schema.ParseGroupVersion(list.GroupVersion)
		if err != nil {
			continue
		}
		for _, resource := range list.APIResources {
			if len(resource.Verbs) == 0 {
				continue
			}
			// filter apiGroup
			if groupChanged && o.APIGroup != gv.Group {
				continue
			}
			// filter namespaced
			if nsChanged && o.Namespaced != resource.Namespaced {
				continue
			}
			// filter to resources that support the specified verbs
			if len(o.Verbs) > 0 && !sets.NewString(resource.Verbs...).HasAll(o.Verbs...) {
				continue
			}
			resources = append(resources, groupResource{
				APIGroup:    gv.Group,
				APIResource: resource,
			})
		}
	}

	if o.NoHeaders == false && o.Output != "name" {
		if err = printContextHeaders(w, o.Output); err != nil {
			return err
		}
	}

	sort.Stable(sortableGroupResource(resources))
	for _, r := range resources {
		switch o.Output {
		case "name":
			name := r.APIResource.Name
			if len(r.APIGroup) > 0 {
				name += "." + r.APIGroup
			}
			if _, err := fmt.Fprintf(w, "%s\n", name); err != nil {
				errs = append(errs, err)
			}
		case "wide":
			if _, err := fmt.Fprintf(w, "%s\t%s\t%s\t%v\t%s\t%v\n",
				r.APIResource.Name,
				strings.Join(r.APIResource.ShortNames, ","),
				r.APIGroup,
				r.APIResource.Namespaced,
				r.APIResource.Kind,
				r.APIResource.Verbs); err != nil {
				errs = append(errs, err)
			}
		case "":
			if _, err := fmt.Fprintf(w, "%s\t%s\t%s\t%v\t%s\n",
				r.APIResource.Name,
				strings.Join(r.APIResource.ShortNames, ","),
				r.APIGroup,
				r.APIResource.Namespaced,
				r.APIResource.Kind); err != nil {
				errs = append(errs, err)
			}
		}
	}

	if len(errs) > 0 {
		return errors.NewAggregate(errs)
	}
	return nil
}

func printContextHeaders(out io.Writer, output string) error {
	columnNames := []string{"NAME", "SHORTNAMES", "APIGROUP", "NAMESPACED", "KIND"}
	if output == "wide" {
		columnNames = append(columnNames, "VERBS")
	}
	_, err := fmt.Fprintf(out, "%s\n", strings.Join(columnNames, "\t"))
	return err
}

type sortableGroupResource []groupResource

func (s sortableGroupResource) Len() int      { return len(s) }
func (s sortableGroupResource) Swap(i, j int) { s[i], s[j] = s[j], s[i] }
func (s sortableGroupResource) Less(i, j int) bool {
	ret := strings.Compare(s[i].APIGroup, s[j].APIGroup)
	if ret > 0 {
		return false
	} else if ret == 0 {
		return strings.Compare(s[i].APIResource.Name, s[j].APIResource.Name) < 0
	}
	return true
}
