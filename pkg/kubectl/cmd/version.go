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
	"io"

	"github.com/ghodss/yaml"
	"github.com/spf13/cobra"

	apimachineryversion "k8s.io/apimachinery/pkg/version"
	"k8s.io/kubernetes/pkg/kubectl/cmd/templates"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/kubectl/cmd/util/cmdproto"
	protofile "k8s.io/kubernetes/pkg/kubectl/cmd/util/cmdproto/k8s_io_kubectl_cmd"
	"k8s.io/kubernetes/pkg/kubectl/util/i18n"
	"k8s.io/kubernetes/pkg/version"
)

var (
	versionExample = templates.Examples(i18n.T(`
		# Print the client and server versions for the current context
		kubectl version`))
)

type Version struct {
	ClientVersion *apimachineryversion.Info `json:"clientVersion,omitempty" yaml:"clientVersion,omitempty"`
	ServerVersion *apimachineryversion.Info `json:"serverVersion,omitempty" yaml:"serverVersion,omitempty"`
}

// VersionOptions: describe the options available to users of the "kubectl
// version" command.
type VersionOptions struct {
	flags protofile.VersionFlags
}

func NewCmdVersion(f cmdutil.Factory, out io.Writer) *cobra.Command {
	options := new(VersionOptions)

	cmd := &cobra.Command{
		Use:     "version",
		Short:   i18n.T("Print the client and server version information"),
		Long:    "Print the client and server version information for the current context",
		Example: versionExample,
		Run: func(cmd *cobra.Command, args []string) {
			cmdutil.CheckErr(options.Complete())
			cmdutil.CheckErr(options.Validate())
			cmdutil.CheckErr(options.Run(f, out))
		},
	}

	cmdproto.FlagsSetup(cmd, &options.flags)

	cmd.Flags().MarkShorthandDeprecated("client", "please use --client instead.")
	return cmd
}

func retrieveServerVersion(f cmdutil.Factory) (*apimachineryversion.Info, error) {
	discoveryClient, err := f.DiscoveryClient()
	if err != nil {
		return nil, err
	}

	// Always request fresh data from the server
	discoveryClient.Invalidate()
	return discoveryClient.ServerVersion()
}

func (o *VersionOptions) Run(f cmdutil.Factory, out io.Writer) error {
	var (
		serverVersion *apimachineryversion.Info
		serverErr     error
		versionInfo   Version
	)

	clientVersion := version.Get()
	versionInfo.ClientVersion = &clientVersion

	if !o.flags.GetClient() {
		serverVersion, serverErr = retrieveServerVersion(f)
		versionInfo.ServerVersion = serverVersion
	}

	switch o.flags.GetOutput() {
	case "":
		if o.flags.GetShort() {
			fmt.Fprintf(out, "Client Version: %s\n", clientVersion.GitVersion)
			if serverVersion != nil {
				fmt.Fprintf(out, "Server Version: %s\n", serverVersion.GitVersion)
			}
		} else {
			fmt.Fprintf(out, "Client Version: %s\n", fmt.Sprintf("%#v", clientVersion))
			if serverVersion != nil {
				fmt.Fprintf(out, "Server Version: %s\n", fmt.Sprintf("%#v", *serverVersion))
			}
		}
	case "yaml":
		marshalled, err := yaml.Marshal(&versionInfo)
		if err != nil {
			return err
		}
		fmt.Fprintln(out, string(marshalled))
	case "json":
		marshalled, err := json.MarshalIndent(&versionInfo, "", "  ")
		if err != nil {
			return err
		}
		fmt.Fprintln(out, string(marshalled))
	default:
		// There is a bug in the program if we hit this case.
		// However, we follow a policy of never panicking.
		return fmt.Errorf("VersionOptions were not validated: --output=%q should have been rejected", o.flags.GetOutput())
	}

	return serverErr
}

func (o *VersionOptions) Complete() error {
	return nil
}

func (o *VersionOptions) Validate() error {
	output := o.flags.GetOutput()
	if output != "" && output != "yaml" && output != "json" {
		return errors.New(`--output must be 'yaml' or 'json'`)
	}

	return nil
}
