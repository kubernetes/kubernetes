// Copyright 2015 CoreOS, Inc.
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
	"strconv"

	"github.com/spf13/cobra"
	"golang.org/x/net/context"
)

// NewCompactionCommand returns the cobra command for "compaction".
func NewCompactionCommand() *cobra.Command {
	return &cobra.Command{
		Use:   "compaction <revision>",
		Short: "Compaction compacts the event history in etcd.",
		Run:   compactionCommandFunc,
	}
}

// compactionCommandFunc executes the "compaction" command.
func compactionCommandFunc(cmd *cobra.Command, args []string) {
	if len(args) != 1 {
		ExitWithError(ExitBadArgs, fmt.Errorf("compaction command needs 1 argument."))
	}

	rev, err := strconv.ParseInt(args[0], 10, 64)
	if err != nil {
		ExitWithError(ExitError, err)
	}

	c := mustClientFromCmd(cmd)
	if cerr := c.Compact(context.TODO(), rev); cerr != nil {
		ExitWithError(ExitError, cerr)
		return
	}
	fmt.Println("compacted revision", rev)
}
