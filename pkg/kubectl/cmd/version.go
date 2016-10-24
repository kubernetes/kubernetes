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
	"io"

	"github.com/spf13/cobra"

	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/version"
)

func NewCmdVersion(f cmdutil.Factory, out io.Writer) *cobra.Command {
	cmd := &cobra.Command{
		Use:   "version",
		Short: "Print the client and server version information",
		Run: func(cmd *cobra.Command, args []string) {
			err := RunVersion(f, out, cmd)
			cmdutil.CheckErr(err)
		},
	}
	cmd.Flags().BoolP("client", "c", false, "Client version only (no server required).")
	cmd.Flags().BoolP("short", "", false, "Print just the version number.")
	cmd.Flags().MarkShorthandDeprecated("client", "please use --client instead.")
	return cmd
}

func RunVersion(f cmdutil.Factory, out io.Writer, cmd *cobra.Command) error {
	v := fmt.Sprintf("%#v", version.Get())
	if cmdutil.GetFlagBool(cmd, "short") {
		v = version.Get().GitVersion
	}

	fmt.Fprintf(out, "Client Version: %s\n", v)
	if cmdutil.GetFlagBool(cmd, "client") {
		return nil
	}

	clientset, err := f.ClientSet()
	if err != nil {
		return err
	}

	serverVersion, err := clientset.Discovery().ServerVersion()
	if err != nil {
		return err
	}

	v = fmt.Sprintf("%#v", *serverVersion)
	if cmdutil.GetFlagBool(cmd, "short") {
		v = serverVersion.GitVersion
	}

	fmt.Fprintf(out, "Server Version: %s\n", v)
	return nil
}
