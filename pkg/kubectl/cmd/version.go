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
	"k8s.io/kubernetes/pkg/util/i18n"
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

func NewCmdVersion(f cmdutil.Factory, out io.Writer) *cobra.Command {
	cmd := &cobra.Command{
		Use:     "version",
		Short:   i18n.T("Print the client and server version information"),
		Long:    "Print the client and server version information for the current context",
		Example: versionExample,
		Run: func(cmd *cobra.Command, args []string) {
			err := RunVersion(f, out, cmd)
			cmdutil.CheckErr(err)
		},
	}
	cmd.Flags().BoolP("client", "c", false, "Client version only (no server required).")
	cmd.Flags().BoolP("short", "", false, "Print just the version number.")
	cmd.Flags().String("output", "", "output format, options available are yaml and json")
	cmd.Flags().MarkShorthandDeprecated("client", "please use --client instead.")
	return cmd
}

func RunVersion(f cmdutil.Factory, out io.Writer, cmd *cobra.Command) error {
	var serverVersion *apimachineryversion.Info = nil
	var serverErr error = nil
	vo := Version{nil, nil}

	clientVersion := version.Get()
	vo.ClientVersion = &clientVersion

	if !cmdutil.GetFlagBool(cmd, "client") {
		serverVersion, serverErr = retrieveServerVersion(f)
		vo.ServerVersion = serverVersion
	}

	switch of := cmdutil.GetFlagString(cmd, "output"); of {
	case "":
		if cmdutil.GetFlagBool(cmd, "short") {
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
		y, err := yaml.Marshal(&vo)
		if err != nil {
			return err
		}

		fmt.Fprintln(out, string(y))
	case "json":
		y, err := json.Marshal(&vo)
		if err != nil {
			return err
		}
		fmt.Fprintln(out, string(y))
	default:
		return errors.New("invalid output format: " + of)

	}

	if serverErr != nil {
		return serverErr
	}

	return nil
}

func retrieveServerVersion(f cmdutil.Factory) (*apimachineryversion.Info, error) {
	discoveryClient, err := f.DiscoveryClient()
	if err != nil {
		return nil, err
	}

	// Always request fresh data from the server
	discoveryClient.Invalidate()

	serverVersion, err := discoveryClient.ServerVersion()
	if err != nil {
		return nil, err
	}

	return serverVersion, nil
}
