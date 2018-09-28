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
	"encoding/json"
	"errors"
	"fmt"

	"github.com/ghodss/yaml"
	"github.com/spf13/cobra"

	apimachineryversion "k8s.io/apimachinery/pkg/version"
	"k8s.io/cli-runtime/pkg/genericclioptions"
	"k8s.io/client-go/discovery"
	"k8s.io/kubernetes/pkg/kubectl/cmd/templates"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/kubectl/util/i18n"
	"k8s.io/kubernetes/pkg/version"
)

type Version struct {
	ClientVersion *apimachineryversion.Info `json:"clientVersion,omitempty" yaml:"clientVersion,omitempty"`
	ServerVersion *apimachineryversion.Info `json:"serverVersion,omitempty" yaml:"serverVersion,omitempty"`
}

var (
	versionExample = templates.Examples(i18n.T(`
		# Print the client and server versions for the current context
		kubectl version`))
)

type VersionOptions struct {
	ClientOnly bool
	Short      bool
	Output     string

	discoveryClient discovery.CachedDiscoveryInterface

	genericclioptions.IOStreams
}

func NewVersionOptions(ioStreams genericclioptions.IOStreams) *VersionOptions {
	return &VersionOptions{
		IOStreams: ioStreams,
	}

}

func NewCmdVersion(f cmdutil.Factory, ioStreams genericclioptions.IOStreams) *cobra.Command {
	o := NewVersionOptions(ioStreams)
	cmd := &cobra.Command{
		Use:     "version",
		Short:   i18n.T("Print the client and server version information"),
		Long:    "Print the client and server version information for the current context",
		Example: versionExample,
		Run: func(cmd *cobra.Command, args []string) {
			cmdutil.CheckErr(o.Complete(f, cmd))
			cmdutil.CheckErr(o.Validate())
			cmdutil.CheckErr(o.Run())
		},
	}
	cmd.Flags().BoolVar(&o.ClientOnly, "client", o.ClientOnly, "Client version only (no server required).")
	cmd.Flags().BoolVar(&o.Short, "short", o.Short, "Print just the version number.")
	cmd.Flags().StringVarP(&o.Output, "output", "o", o.Output, "One of 'yaml' or 'json'.")
	return cmd
}

func (o *VersionOptions) Complete(f cmdutil.Factory, cmd *cobra.Command) error {
	var err error
	o.discoveryClient, err = f.ToDiscoveryClient()
	if err != nil {
		return err
	}
	return nil
}

func (o *VersionOptions) Validate() error {
	if o.Output != "" && o.Output != "yaml" && o.Output != "json" {
		return errors.New(`--output must be 'yaml' or 'json'`)
	}

	return nil
}

func (o *VersionOptions) Run() error {
	var (
		serverVersion *apimachineryversion.Info
		serverErr     error
		versionInfo   Version
	)

	clientVersion := version.Get()
	versionInfo.ClientVersion = &clientVersion

	if !o.ClientOnly {
		// Always request fresh data from the server
		o.discoveryClient.Invalidate()
		serverVersion, serverErr = o.discoveryClient.ServerVersion()
		versionInfo.ServerVersion = serverVersion
	}

	switch o.Output {
	case "":
		if o.Short {
			fmt.Fprintf(o.Out, "Client Version: %s\n", clientVersion.GitVersion)
			if serverVersion != nil {
				fmt.Fprintf(o.Out, "Server Version: %s\n", serverVersion.GitVersion)
			}
		} else {
			fmt.Fprintf(o.Out, "Client Version: %s\n", fmt.Sprintf("%#v", clientVersion))
			if serverVersion != nil {
				fmt.Fprintf(o.Out, "Server Version: %s\n", fmt.Sprintf("%#v", *serverVersion))
			}
		}
	case "yaml":
		marshalled, err := yaml.Marshal(&versionInfo)
		if err != nil {
			return err
		}
		fmt.Fprintln(o.Out, string(marshalled))
	case "json":
		marshalled, err := json.MarshalIndent(&versionInfo, "", "  ")
		if err != nil {
			return err
		}
		fmt.Fprintln(o.Out, string(marshalled))
	default:
		// There is a bug in the program if we hit this case.
		// However, we follow a policy of never panicking.
		return fmt.Errorf("VersionOptions were not validated: --output=%q should have been rejected", o.Output)
	}

	return serverErr
}
