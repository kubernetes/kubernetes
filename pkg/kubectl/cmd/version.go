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
	"io"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/kubectl"
	"github.com/spf13/cobra"
)

func NewCmdVersion(out io.Writer) *cobra.Command {
	cmd := &cobra.Command{
		Use:   "version",
		Short: "Print version of client and server",
		Run: func(cmd *cobra.Command, args []string) {
			if GetFlagBool(cmd, "client") {
				kubectl.GetClientVersion(out)
			} else {
				kubectl.GetVersion(out, getKubeClient(cmd))
			}
		},
	}
	cmd.Flags().BoolP("client", "c", false, "Client version only (no server required)")
	return cmd
}
