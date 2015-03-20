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
		Use:   "describe RESOURCE ID",
		Short: "Show details of a specific resource",
		Long: `Show details of a specific resource.

This command joins many API calls together to form a detailed description of a
given resource.`,
		Run: func(cmd *cobra.Command, args []string) {
			err := RunDescribe(f, out, cmd, args)
			util.CheckErr(err)
		},
	}
	return cmd
}

func RunDescribe(f *Factory, out io.Writer, cmd *cobra.Command, args []string) error {
	cmdNamespace, err := f.DefaultNamespace(cmd)
	if err != nil {
		return err
	}

	mapper, _ := f.Object(cmd)
	// TODO: use resource.Builder instead
	mapping, namespace, name, err := util.ResourceFromArgs(cmd, args, mapper, cmdNamespace)
	if err != nil {
		return err
	}

	describer, err := f.Describer(cmd, mapping)
	if err != nil {
		return err
	}

	s, err := describer.Describe(namespace, name)
	if err != nil {
		return err
	}
	fmt.Fprintf(out, "%s\n", s)
	return nil
}
