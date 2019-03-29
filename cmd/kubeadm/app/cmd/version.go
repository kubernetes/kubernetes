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
	"fmt"
	"io"

	"github.com/spf13/cobra"
	apimachineryversion "k8s.io/apimachinery/pkg/version"
	"k8s.io/klog"

	"k8s.io/kubernetes/cmd/kubeadm/app/cmd/options"
	kubeadmutil "k8s.io/kubernetes/cmd/kubeadm/app/util"
	"k8s.io/kubernetes/pkg/version"
)

// Version provides the version information of kubeadm.
type Version struct {
	ClientVersion *apimachineryversion.Info `json:"clientVersion"`
}

func (version *Version) Short() string {
	return version.ClientVersion.GitVersion
}

func (version *Version) Text() string {
	return fmt.Sprintf("kubeadm version: %#v", version)
}

// NewCmdVersion provides the version information of kubeadm.
func NewCmdVersion(out io.Writer) *cobra.Command {
	var outputFormat string
	cmd := &cobra.Command{
		Use:   "version",
		Short: "Print the version of kubeadm",
		Run: func(cmd *cobra.Command, args []string) {
			err := RunVersion(out, cmd, outputFormat)
			kubeadmutil.CheckErr(err)
		},
	}
	options.AddOutputFlag(cmd.Flags(), &outputFormat)
	return cmd
}

// RunVersion provides the version information of kubeadm in format depending on arguments
// specified in cobra.Command.
func RunVersion(out io.Writer, cmd *cobra.Command, outputFormat string) error {
	klog.V(1).Infoln("[version] retrieving version info")
	clientVersion := version.Get()
	v := Version{ClientVersion: &clientVersion}

	output, err := kubeadmutil.ConvertToOutputFormat(&v, outputFormat)
	if err != nil {
		return err
	}

	fmt.Fprintln(out, output)

	return nil
}
