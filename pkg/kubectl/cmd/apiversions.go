/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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
	"os"

	"github.com/spf13/cobra"

	"k8s.io/kubernetes/pkg/kubectl"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
)

func NewCmdApiVersions(f *cmdutil.Factory, out io.Writer) *cobra.Command {
	cmd := &cobra.Command{
		Use: "api-versions",
		// apiversions is deprecated.
		Aliases: []string{"apiversions"},
		Short:   "Print available API versions.",
		Run: func(cmd *cobra.Command, args []string) {
			err := RunApiVersions(f, out)
			cmdutil.CheckErr(err)
		},
	}
	return cmd
}

func RunApiVersions(f *cmdutil.Factory, out io.Writer) error {
	if len(os.Args) > 1 && os.Args[1] == "apiversions" {
		printDeprecationWarning("api-versions", "apiversions")
	}

	client, err := f.Client()
	if err != nil {
		return err
	}

	kubectl.GetApiVersions(out, client)
	return nil
}
