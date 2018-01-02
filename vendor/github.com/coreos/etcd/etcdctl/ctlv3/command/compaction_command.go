// Copyright 2015 The etcd Authors
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

	"github.com/coreos/etcd/clientv3"
	"github.com/spf13/cobra"
)

var compactPhysical bool

// NewCompactionCommand returns the cobra command for "compaction".
func NewCompactionCommand() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "compaction [options] <revision>",
		Short: "Compacts the event history in etcd",
		Run:   compactionCommandFunc,
	}
	cmd.Flags().BoolVar(&compactPhysical, "physical", false, "'true' to wait for compaction to physically remove all old revisions")
	return cmd
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

	var opts []clientv3.CompactOption
	if compactPhysical {
		opts = append(opts, clientv3.WithCompactPhysical())
	}

	c := mustClientFromCmd(cmd)
	ctx, cancel := commandCtx(cmd)
	_, cerr := c.Compact(ctx, rev, opts...)
	cancel()
	if cerr != nil {
		ExitWithError(ExitError, cerr)
		return
	}
	fmt.Println("compacted revision", rev)
}
