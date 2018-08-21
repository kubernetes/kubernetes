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
	"sort"

	"github.com/spf13/cobra"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/cli-runtime/pkg/genericclioptions"
	"k8s.io/client-go/discovery"
	"k8s.io/kubernetes/pkg/kubectl/cmd/templates"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/kubectl/util/i18n"
)

var (
	apiversionsExample = templates.Examples(i18n.T(`
		# Print the supported API versions
		kubectl api-versions`))
)

type ApiVersionsOptions struct {
	discoveryClient discovery.CachedDiscoveryInterface

	genericclioptions.IOStreams
}

func NewApiVersionsOptions(ioStreams genericclioptions.IOStreams) *ApiVersionsOptions {
	return &ApiVersionsOptions{
		IOStreams: ioStreams,
	}
}

func NewCmdApiVersions(f cmdutil.Factory, ioStreams genericclioptions.IOStreams) *cobra.Command {
	o := NewApiVersionsOptions(ioStreams)
	cmd := &cobra.Command{
		Use:     "api-versions",
		Short:   "Print the supported API versions on the server, in the form of \"group/version\"",
		Long:    "Print the supported API versions on the server, in the form of \"group/version\"",
		Example: apiversionsExample,
		Run: func(cmd *cobra.Command, args []string) {
			cmdutil.CheckErr(o.Complete(f))
			cmdutil.CheckErr(o.RunApiVersions())
		},
	}
	return cmd
}

func (o *ApiVersionsOptions) Complete(f cmdutil.Factory) error {
	var err error
	o.discoveryClient, err = f.ToDiscoveryClient()
	if err != nil {
		return err
	}
	return nil
}

func (o *ApiVersionsOptions) RunApiVersions() error {
	// Always request fresh data from the server
	o.discoveryClient.Invalidate()

	groupList, err := o.discoveryClient.ServerGroups()
	if err != nil {
		return fmt.Errorf("Couldn't get available api versions from server: %v\n", err)
	}
	apiVersions := metav1.ExtractGroupVersions(groupList)
	sort.Strings(apiVersions)
	for _, v := range apiVersions {
		fmt.Fprintln(o.Out, v)
	}
	return nil
}
