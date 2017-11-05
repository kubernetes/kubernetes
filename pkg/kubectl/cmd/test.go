/*
Copyright 2017 The Kubernetes Authors.

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
	"io"

	"github.com/spf13/cobra"

	apimachineryversion "k8s.io/apimachinery/pkg/version"
	"k8s.io/kubernetes/pkg/kubectl/cmd/templates"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/kubectl/cmd/util/cmdproto"
	protofile "k8s.io/kubernetes/pkg/kubectl/cmd/util/cmdproto/k8s_io_kubectl_cmd"
	"k8s.io/kubernetes/pkg/kubectl/util/i18n"
)

var (
	testExample = templates.Examples(i18n.T(`
		# Print the client and server versions for the current context
		kubectl version`))
)

type Test struct {
	ClientVersion *apimachineryversion.Info `json:"clientVersion,omitempty" yaml:"clientVersion,omitempty"`
	ServerVersion *apimachineryversion.Info `json:"serverVersion,omitempty" yaml:"serverVersion,omitempty"`
}

// TestOptions
type TestOptions struct {
	flags protofile.TestCmd
}

func NewCmdTest(f cmdutil.Factory, out io.Writer) *cobra.Command {
	options := new(TestOptions)

	cmd := &cobra.Command{
		Use:     "version",
		Short:   i18n.T("Print the client and server version information"),
		Long:    "Print the client and server version information for the current context",
		Example: testExample,
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

func (o *TestOptions) Run(f cmdutil.Factory, out io.Writer) error {
	return nil
}

func (o *TestOptions) Complete() error {
	return nil
}

func (o *TestOptions) Validate() error {
	return nil
}
