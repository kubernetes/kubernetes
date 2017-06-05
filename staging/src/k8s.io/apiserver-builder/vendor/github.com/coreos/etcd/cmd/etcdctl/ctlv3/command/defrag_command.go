// Copyright 2016 The etcd Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package command

import (
	"fmt"
	"os"

	"github.com/spf13/cobra"
)

// NewDefragCommand returns the cobra command for "Defrag".
func NewDefragCommand() *cobra.Command {
	return &cobra.Command{
		Use:   "defrag",
		Short: "Defragments the storage of the etcd members with given endpoints",
		Run:   defragCommandFunc,
	}
}

func defragCommandFunc(cmd *cobra.Command, args []string) {
	failures := 0
	c := mustClientFromCmd(cmd)
	for _, ep := range c.Endpoints() {
		ctx, cancel := commandCtx(cmd)
		_, err := c.Defragment(ctx, ep)
		cancel()
		if err != nil {
			fmt.Fprintf(os.Stderr, "Failed to defragment etcd member[%s] (%v)\n", ep, err)
			failures++
		} else {
			fmt.Printf("Finished defragmenting etcd member[%s]\n", ep)
		}
	}

	if failures != 0 {
		os.Exit(ExitError)
	}
}
