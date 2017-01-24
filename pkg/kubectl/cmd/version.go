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
	"fmt"
	"io"

	"github.com/spf13/cobra"

	"k8s.io/kubernetes/pkg/kubectl/cmd/templates"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/version"
)

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
	cmd.Flags().BoolP("json", "", false, "Print version info in JSON format.")
	cmd.Flags().BoolP("short", "", false, "Print just the version number.")
	cmd.Flags().MarkShorthandDeprecated("client", "please use --client instead.")
	return cmd
}

type versionInfo struct {
	Client version.Info  `json:"client"`
	Server *version.Info `json:"server,omitempty"`
}

func RunVersion(f cmdutil.Factory, out io.Writer, cmd *cobra.Command) error {
	if cmdutil.GetFlagBool(cmd, "json") && cmdutil.GetFlagBool(cmd, "short") {
		return fmt.Errorf("Cannot use --json and --short!")
	}
	v := versionInfo{Client: version.Get(), Server: nil}

	var serverErr error
	if !cmdutil.GetFlagBool(cmd, "client") {
		clientset, err := f.ClientSet()
		serverErr = err
		if serverErr == nil {
			(v.Server), serverErr = clientset.Discovery().ServerVersion()
		}
	}

	if cmdutil.GetFlagBool(cmd, "json") {
		jsonOut, err := json.MarshalIndent(v, "", "  ")
		if err != nil {
			return err
		}
		fmt.Fprintf(out, "%s\n", jsonOut)
		return serverErr
	}

	verStr := func(v version.Info) string {
		if cmdutil.GetFlagBool(cmd, "short") {
			return v.GitVersion
		}
		return fmt.Sprintf("%#v", v)
	}
	fmt.Fprintf(out, "Client Version: %s\n", verStr(v.Client))
	if serverErr != nil {
		return serverErr
	}
	if v.Server != nil {
		fmt.Fprintf(out, "Server Version: %s\n", verStr(*v.Server))
	}
	return nil
}
