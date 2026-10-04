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

package apiresources

import (
	"fmt"
	"sort"

	"github.com/spf13/cobra"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/cli-runtime/pkg/genericclioptions"
	"k8s.io/cli-runtime/pkg/genericiooptions"
	"k8s.io/client-go/discovery"
	cmdutil "k8s.io/kubectl/pkg/cmd/util"
	"k8s.io/kubectl/pkg/util/i18n"
	"k8s.io/kubectl/pkg/util/templates"
)

var (
	apiversionsExample = templates.Examples(i18n.T(`
		# Print the supported API versions
		kubectl api-versions`))
)

// APIVersionsFlags directly reflect the information that CLI is gathering via flags
type APIVersionsFlags struct {
	RESTClientGetter genericclioptions.RESTClientGetter

	genericiooptions.IOStreams
}

// NewAPIVersionsFlags returns a default APIVersionsFlags
func NewAPIVersionsFlags(restClientGetter genericclioptions.RESTClientGetter, ioStreams genericiooptions.IOStreams) *APIVersionsFlags {
	return &APIVersionsFlags{
		RESTClientGetter: restClientGetter,
		IOStreams:        ioStreams,
	}
}

// APIVersionsOptions is the start of the data required to perform the operation. As new fields are added,
// add them here instead of referencing the cmd.Flags()
type APIVersionsOptions struct {
	discoveryClient discovery.CachedDiscoveryInterface

	genericiooptions.IOStreams
}

// NewCmdAPIVersions creates the `api-versions` command
func NewCmdAPIVersions(restClientGetter genericclioptions.RESTClientGetter, ioStreams genericiooptions.IOStreams) *cobra.Command {
	flags := NewAPIVersionsFlags(restClientGetter, ioStreams)
	cmd := &cobra.Command{
		Use:                   "api-versions",
		Short:                 i18n.T("Print the supported API versions on the server, in the form of \"group/version\""),
		Long:                  i18n.T("Print the supported API versions on the server, in the form of \"group/version\"."),
		Example:               apiversionsExample,
		DisableFlagsInUseLine: true,
		Run: func(cmd *cobra.Command, args []string) {
			o, err := flags.ToOptions(args)
			cmdutil.CheckErr(err)
			cmdutil.CheckErr(o.RunAPIVersions())
		},
	}
	return cmd
}

// ToOptions converts from CLI inputs to runtime inputs
func (flags *APIVersionsFlags) ToOptions(args []string) (*APIVersionsOptions, error) {
	if len(args) != 0 {
		return nil, fmt.Errorf("unexpected arguments: %v", args)
	}
	discoveryClient, err := flags.RESTClientGetter.ToDiscoveryClient()
	if err != nil {
		return nil, err
	}
	return &APIVersionsOptions{
		discoveryClient: discoveryClient,
		IOStreams:       flags.IOStreams,
	}, nil
}

// RunAPIVersions does the work
func (o *APIVersionsOptions) RunAPIVersions() error {
	// Always request fresh data from the server
	o.discoveryClient.Invalidate()

	groupList, err := o.discoveryClient.ServerGroups()
	if err != nil {
		return fmt.Errorf("couldn't get available api versions from server: %v", err)
	}
	apiVersions := metav1.ExtractGroupVersions(groupList)
	sort.Strings(apiVersions)
	for _, v := range apiVersions {
		fmt.Fprintln(o.Out, v)
	}
	return nil
}
