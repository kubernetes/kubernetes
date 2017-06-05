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

package set

import (
	"io"

	"github.com/spf13/cobra"
	"k8s.io/kubernetes/pkg/kubectl/cmd/templates"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/util/i18n"
)

var (
	set_long = templates.LongDesc(`
		Configure application resources

		These commands help you make changes to existing application resources.`)
)

func NewCmdSet(f cmdutil.Factory, out, err io.Writer) *cobra.Command {
	cmd := &cobra.Command{
		Use:   "set SUBCOMMAND",
		Short: i18n.T("Set specific features on objects"),
		Long:  set_long,
		Run:   cmdutil.DefaultSubCommandRun(err),
	}

	// add subcommands
	cmd.AddCommand(NewCmdImage(f, out, err))
	cmd.AddCommand(NewCmdResources(f, out, err))
	cmd.AddCommand(NewCmdSelector(f, out))

	return cmd
}
