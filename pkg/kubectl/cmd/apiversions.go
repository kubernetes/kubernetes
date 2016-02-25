/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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
	"errors"
	"fmt"
	"io"
	"os"
	"sort"
	"strings"

	"github.com/spf13/cobra"

	unversioned_client "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/kubectl"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	utilerrors "k8s.io/kubernetes/pkg/util/errors"
)

// ApiVersionsOptions contains all the things necessary to run the api-versions command.
type ApiVersionsOptions struct {
	discovery        unversioned_client.DiscoveryInterface
	out              io.Writer
	noHeaders        bool
	showResources    bool
	onlyNamespaced   bool
	showSubresources bool
}

const (
	apiVersionsLong = `Print the supported API versions and/or resources on the server.

When run without flags, this command prints out API versions in the form of
"group/version".

When run with --show-resources, this command prints out all resources in all
API versions.`

	apiVersionsExample = `# Print all API versions
$ kubectl api-versions

# Print all resources for all API versions, showing only namespaced resources,
# including subresources.
$ kubectl api-versions --show-resources --only-namespaced --show-subresources`
)

func NewCmdApiVersions(f *cmdutil.Factory, out io.Writer) *cobra.Command {
	options := &ApiVersionsOptions{}

	cmd := &cobra.Command{
		Use: "api-versions",
		// apiversions is deprecated.
		Aliases: []string{"apiversions"},
		Short:   "Print the supported API versions and/or resources on the server, in the form of \"group/version\".",
		Long:    apiVersionsLong,
		Example: apiVersionsExample,
		Run: func(cmd *cobra.Command, args []string) {
			cmdutil.CheckErr(options.Complete(f, cmd, args, out))
			cmdutil.CheckErr(options.Validate())
			cmdutil.CheckErr(options.Run())
		},
	}

	cmd.Flags().BoolVar(&options.showResources, "show-resources", options.showResources, "Show resources in each API version")
	cmd.Flags().BoolVar(&options.noHeaders, "no-headers", options.noHeaders, "When using the default output, don't print headers.")
	cmd.Flags().BoolVar(&options.onlyNamespaced, "only-namespaced", options.onlyNamespaced, "Only show namespaced resources")
	cmd.Flags().BoolVar(&options.showSubresources, "show-subresources", options.showSubresources, "Show subresources")

	return cmd
}

// Complete completes all the required options for api-versions.
func (o *ApiVersionsOptions) Complete(f *cmdutil.Factory, cmd *cobra.Command, args []string, out io.Writer) error {
	if len(os.Args) > 1 && os.Args[1] == "apiversions" {
		printDeprecationWarning("api-versions", "apiversions")
	}

	client, err := f.Client()
	if err != nil {
		return err
	}

	o.discovery = client.Discovery()
	o.out = out

	return nil
}

// Validate validates all the required options for api-versions.
func (o *ApiVersionsOptions) Validate() error {
	if o.discovery == nil {
		return errors.New("discovery interface must be set")
	}

	if o.out == nil {
		return errors.New("out must be set")
	}

	return nil
}

// Run prints out the resources for the specified api version.
func (o *ApiVersionsOptions) Run() error {
	groupList, err := o.discovery.ServerGroups()
	if err != nil {
		return err
	}
	apiVersions := unversioned_client.ExtractGroupVersions(groupList)
	sort.Strings(apiVersions)

	w := kubectl.GetNewTabWriter(o.out)
	defer w.Flush()

	if o.showResources {
		return o.printResources(apiVersions, w)
	}

	printGroupVersions(apiVersions, w)
	return nil
}

func printGroupVersions(apiVersions []string, w io.Writer) {
	for _, v := range apiVersions {
		fmt.Fprintln(w, v)
	}
}

func (o *ApiVersionsOptions) printResources(apiVersions []string, w io.Writer) error {
	errs := []error{}

	if !o.noHeaders {
		fmt.Fprintln(w, "GROUP/VERSION\tRESOURCE\tNAMESPACED")
	}
	for _, groupVersion := range apiVersions {
		resources, err := o.discovery.ServerResourcesForGroupVersion(groupVersion)
		if err != nil {
			errs = append(errs, err)
		}

		for _, r := range resources.APIResources {
			if o.onlyNamespaced && !r.Namespaced {
				continue
			}
			if !o.showSubresources && strings.Contains(r.Name, "/") {
				continue
			}
			fmt.Fprintf(w, "%s\t%s\t%t\n", groupVersion, r.Name, r.Namespaced)
		}
	}

	return utilerrors.NewAggregate(errs)
}
