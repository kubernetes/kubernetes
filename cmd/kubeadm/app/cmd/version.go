/*
Copyright 2016 The Kubernetes Authors.

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

	"github.com/ghodss/yaml"
	"github.com/spf13/cobra"

	apimachineryversion "k8s.io/apimachinery/pkg/version"
	kubeadmutil "k8s.io/kubernetes/cmd/kubeadm/app/util"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/version"
)

type Version struct {
	// clientVersion contains the kubeadm version.
	clientVersion *apimachineryversion.Info `json:"kubeadmVersion,omitempty" yaml:"kubeadmVersion,omitempty"`
}

func NewCmdVersion(out io.Writer) *cobra.Command {
	cmd := &cobra.Command{
		Use:   "version",
		Short: "Print the version of kubeadm",
		Run: func(cmd *cobra.Command, args []string) {
			err := RunVersion(out, cmd)
			kubeadmutil.CheckErr(err)
		},
	}
	cmd.Flags().BoolP("short", "", false, "only print the version (tag|number)")
	cmd.Flags().StringP("output", "o", "", "output format, options available are yaml and json")
	return cmd
}

func RunVersion(out io.Writer, cmd *cobra.Command) error {

	v := Version{
		clientVersion: &version.Get(),
	}

	switch of := cmdutil.GetFlagString(cmd, "output"); of {
	case "":
		if cmdutil.GetFlagBool(cmd, "short") {
			fmt.Fprintf(out, "kubeadm version: %s\n", v.clientVersion.GitVersion)
		} else {
			fmt.Fprintf(out, "kubeadm version: %#v\n", *v.clientVersion)
		}
	case "yaml":
		y, err := yaml.Marshal(&v)
		if err != nil {
			return err
		}
		fmt.Fprintln(out, string(y))
	case "json":
		y, err := json.Marshal(&v)
		if err != nil {
			return err
		}
		fmt.Fprintln(out, string(y))
	default:
		return fmt.Errorf("invalid output format: %s", of)
	}

	return nil
}
