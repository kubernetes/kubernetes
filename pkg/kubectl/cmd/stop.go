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

	"github.com/spf13/cobra"
)

func (f *Factory) NewCmdStop(out io.Writer) *cobra.Command {
	cmd := &cobra.Command{
		Use:   "stop <resource> <id>",
		Short: "Gracefully shutdown a resource",
		Long: `Gracefully shutdown a resource

Attempts to shutdown and delete a resource that supports graceful termination.
If the resource is resizable it will be resized to 0 before deletion.

Examples:
  $ kubectl stop replicationcontroller foo
  foo stopped
`,
		Run: func(cmd *cobra.Command, args []string) {
			if len(args) != 2 {
				usageError(cmd, "<resource> <id>")
			}
			mapper, _ := f.Object(cmd)
			mapping, namespace, name := ResourceFromArgs(cmd, args, mapper)

			reaper, err := f.Reaper(cmd, mapping)
			checkErr(err)

			s, err := reaper.Stop(namespace, name)
			checkErr(err)
			fmt.Fprintf(out, "%s\n", s)
		},
	}
	return cmd
}
