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
	"errors"
	"fmt"
	"github.com/spf13/cobra"
	"io"

	"encoding/json"
	"github.com/ghodss/yaml"
	"k8s.io/kubernetes/pkg/kubectl/cmd/templates"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/version"
)

type VersionObj struct {
	ClientVersion *version.Info `json:"clientVersion,omitempty" yaml:"clientVersion,omitempty"`
	ServerVersion *version.Info `json:"serverVersion,omitempty" yaml:"serverVersion,omitempty"`
}

var (
	version_example = templates.Examples(`
		# Print the client and server versions for the current context
		kubectl version`)
)

func NewCmdVersion(f cmdutil.Factory, out io.Writer) *cobra.Command {
	cmd := &cobra.Command{
		Use:     "version",
		Short:   "Print the client and server version information",
		Long:    "Print the client and server version information for the current context",
		Example: version_example,
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
	vo := VersionObj{nil, nil}

	cvg := version.Get()
	vo.ClientVersion = &cvg

	switch of := cmdutil.GetFlagString(cmd, "output"); of {
	case "":
		if cmdutil.GetFlagBool(cmd, "short") {
			fmt.Fprintf(out, "Client Version: %s\n", vo.ClientVersion.GitVersion)
			if cmdutil.GetFlagBool(cmd, "client") {
				return nil
			}

			serverVersion, err := retrieveServerVersion(f)
			if err != nil {
				return err
			}

			fmt.Fprintf(out, "Server Version: %s\n", serverVersion.GitVersion)

		} else {
			fmt.Fprintf(out, "Client Version: %s\n", fmt.Sprintf("%#v", *vo.ClientVersion))
			if cmdutil.GetFlagBool(cmd, "client") {
				return nil
			}

			serverVersion, err := retrieveServerVersion(f)
			if err != nil {
				return err
			}

			fmt.Fprintf(out, "Server Version: %s\n", fmt.Sprintf("%#v", *serverVersion))
		}

		return nil
	case "yaml":
		var serverErr error = nil
		var serverVersion *version.Info = nil

		if !cmdutil.GetFlagBool(cmd, "client") {
			serverVersion, serverErr = retrieveServerVersion(f)
			vo.ServerVersion = serverVersion
		}

		y, err := yaml.Marshal(&vo)
		if err != nil {
			return err
		}

		fmt.Fprintln(out, string(y))

		if serverErr != nil {
			return serverErr
		}

		return nil
	case "json":
		var serverErr error = nil
		var serverVersion *version.Info = nil

		if !cmdutil.GetFlagBool(cmd, "client") {
			serverVersion, serverErr = retrieveServerVersion(f)
			vo.ServerVersion = serverVersion
		}

		y, err := json.Marshal(&vo)
		if err != nil {
			return err
		}
		fmt.Fprintln(out, string(y))

		if serverErr != nil {
			return serverErr
		}

		return nil
	default:
		return errors.New("invalid output format: " + of)

	}
}

func retrieveServerVersion(f cmdutil.Factory) (*version.Info, error) {
	clientSet, err := f.ClientSet()
	if err != nil {
		return nil, err
	}

	serverVersion, err := clientSet.Discovery().ServerVersion()
	if err != nil {
		return nil, err
	}

	return serverVersion, nil
}
