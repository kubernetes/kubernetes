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

// We include both struct objects because json marshal does not handle the idea
// of empty struct objects well and will include and empty server object in results
type ClientVersionObj struct {
	ClientVersion version.Info `json:"Client Version" yaml:"Client Version"`
}

type ClientAndServerVersionObj struct {
	ClientVersion version.Info `json:"Client Version" yaml:"Client Version"`
	ServerVersion version.Info `json:"Server Version" yaml:"Server Version"`
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
	// default behavior is std.
	cmd.Flags().String("output", "std", "output format, options available are yaml and json")
	cmd.Flags().MarkShorthandDeprecated("client", "please use --client instead.")
	return cmd
}

func RunVersion(f cmdutil.Factory, out io.Writer, cmd *cobra.Command) error {
	of := cmdutil.GetFlagString(cmd, "output")
	if of != "yaml" && of != "json" && of != "std" {
		return errors.New("invalid output format")
	}

	if of == "std" {
		v := fmt.Sprintf("%#v", version.Get())
		if cmdutil.GetFlagBool(cmd, "short") {
			v = version.Get().GitVersion
		}

		fmt.Fprintf(out, "Client Version: %s\n", v)
		if cmdutil.GetFlagBool(cmd, "client") {
			return nil
		}

		clientSet, err := f.ClientSet()
		if err != nil {
			return err
		}

		serverVersion, err := clientSet.Discovery().ServerVersion()
		if err != nil {
			return err
		}

		v = fmt.Sprintf("%#v", *serverVersion)
		if cmdutil.GetFlagBool(cmd, "short") {
			v = serverVersion.GitVersion
		}

		fmt.Fprintf(out, "Server Version: %s\n", v)
		return nil
	} else {
		cv := version.Get()
		if cmdutil.GetFlagBool(cmd, "short") {
			cv = version.Info{GitVersion: version.Get().GitVersion}
		}

		if cmdutil.GetFlagBool(cmd, "client") {
			return printRightFormat(out, of, ClientVersionObj{cv})
		}

		clientSet, err := f.ClientSet()
		if err != nil {
			return err
		}

		serverVersion, err := clientSet.Discovery().ServerVersion()
		if err != nil {
			return err
		}

		sv := *serverVersion
		if cmdutil.GetFlagBool(cmd, "short") {
			sv = version.Info{GitVersion: serverVersion.GitVersion}
		}

		return printRightFormat(out, of, ClientAndServerVersionObj{cv, sv})
	}
}

func printRightFormat(out io.Writer, outputFormat string, vo interface{}) error {
	if outputFormat == "yaml" {
		fmt.Println(vo)
		y, err := yaml.Marshal(&vo)
		if err != nil {
			return err
		}
		fmt.Fprintf(out, string(y))
	} else if outputFormat == "json" {
		y, err := json.Marshal(&vo)
		if err != nil {
			return err
		}
		fmt.Fprintf(out, string(y))
	} else {
		return errors.New("unexpected output format!")
	}

	return nil
}
