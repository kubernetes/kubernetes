/*
Copyright 2014 Google Inc. All rights reserved.

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

	"github.com/GoogleCloudPlatform/kubernetes/pkg/kubectl/cmd/util"
	"github.com/spf13/cobra"
)

func (f *Factory) NewCmdDescribe(out io.Writer) *cobra.Command {
	cmd := &cobra.Command{
		Use:   "describe <resource> <id>",
		Short: "Show details of a specific resource",
		Long: `Show details of a specific resource.

This command joins many API calls together to form a detailed description of a
given resource.`,
		Run: func(cmd *cobra.Command, args []string) {
			cmdNamespace, err := f.DefaultNamespace(cmd)
			util.CheckErr(err)

			mapper, _ := f.Object(cmd)
			// TODO: use resource.Builder instead
			mapping, namespace, name := util.ResourceFromArgs(cmd, args, mapper, cmdNamespace)

			describer, err := f.Describer(cmd, mapping)
			util.CheckErr(err)

			s, err := describer.Describe(namespace, name)
			util.CheckErr(err)
			fmt.Fprintf(out, "%s\n", s)
		},
	}
	return cmd
}
