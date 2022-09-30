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

// APIVersionsOptions have the data required for API versions
type APIVersionsOptions struct {
	discoveryClient discovery.CachedDiscoveryInterface

	genericclioptions.IOStreams
}

// NewAPIVersionsOptions creates the options for APIVersions
func NewAPIVersionsOptions(ioStreams genericclioptions.IOStreams) *APIVersionsOptions {
	return &APIVersionsOptions{
		IOStreams: ioStreams,
	}
}

// NewCmdAPIVersions creates the `api-versions` command
func NewCmdAPIVersions(f cmdutil.Factory, ioStreams genericclioptions.IOStreams) *cobra.Command {
	o := NewAPIVersionsOptions(ioStreams)
	cmd := &cobra.Command{
		Use:                   "api-versions",
		Short:                 i18n.T("Print the supported API versions on the server, in the form of \"group/version\""),
		Long:                  i18n.T("Print the supported API versions on the server, in the form of \"group/version\"."),
		Example:               apiversionsExample,
		DisableFlagsInUseLine: true,
		Run: func(cmd *cobra.Command, args []string) {
			cmdutil.CheckErr(o.Complete(f, cmd, args))
			cmdutil.CheckErr(o.RunAPIVersions())
		},
	}
	return cmd
}

// Complete adapts from the command line args and factory to the data required
func (o *APIVersionsOptions) Complete(f cmdutil.Factory, cmd *cobra.Command, args []string) error {
	if len(args) != 0 {
		return cmdutil.UsageErrorf(cmd, "unexpected arguments: %v", args)
	}
	var err error
	o.discoveryClient, err = f.ToDiscoveryClient()
	return err
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
