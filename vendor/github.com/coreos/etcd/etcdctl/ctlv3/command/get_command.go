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
	"strings"

	"github.com/coreos/etcd/clientv3"
	"github.com/spf13/cobra"
)

var (
	getConsistency string
	getLimit       int64
	getSortOrder   string
	getSortTarget  string
	getPrefix      bool
	getFromKey     bool
	getRev         int64
	getKeysOnly    bool
	printValueOnly bool
)

// NewGetCommand returns the cobra command for "get".
func NewGetCommand() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "get [options] <key> [range_end]",
		Short: "Gets the key or a range of keys",
		Run:   getCommandFunc,
	}

	cmd.Flags().StringVar(&getConsistency, "consistency", "l", "Linearizable(l) or Serializable(s)")
	cmd.Flags().StringVar(&getSortOrder, "order", "", "Order of results; ASCEND or DESCEND (ASCEND by default)")
	cmd.Flags().StringVar(&getSortTarget, "sort-by", "", "Sort target; CREATE, KEY, MODIFY, VALUE, or VERSION")
	cmd.Flags().Int64Var(&getLimit, "limit", 0, "Maximum number of results")
	cmd.Flags().BoolVar(&getPrefix, "prefix", false, "Get keys with matching prefix")
	cmd.Flags().BoolVar(&getFromKey, "from-key", false, "Get keys that are greater than or equal to the given key using byte compare")
	cmd.Flags().Int64Var(&getRev, "rev", 0, "Specify the kv revision")
	cmd.Flags().BoolVar(&getKeysOnly, "keys-only", false, "Get only the keys")
	cmd.Flags().BoolVar(&printValueOnly, "print-value-only", false, `Only write values when using the "simple" output format`)
	return cmd
}

// getCommandFunc executes the "get" command.
func getCommandFunc(cmd *cobra.Command, args []string) {
	key, opts := getGetOp(cmd, args)
	ctx, cancel := commandCtx(cmd)
	resp, err := mustClientFromCmd(cmd).Get(ctx, key, opts...)
	cancel()
	if err != nil {
		ExitWithError(ExitError, err)
	}

	if printValueOnly {
		dp, simple := (display).(*simplePrinter)
		if !simple {
			ExitWithError(ExitBadArgs, fmt.Errorf("print-value-only is only for `--write-out=simple`."))
		}
		dp.valueOnly = true
	}
	display.Get(*resp)
}

func getGetOp(cmd *cobra.Command, args []string) (string, []clientv3.OpOption) {
	if len(args) == 0 {
		ExitWithError(ExitBadArgs, fmt.Errorf("range command needs arguments."))
	}

	if getPrefix && getFromKey {
		ExitWithError(ExitBadArgs, fmt.Errorf("`--prefix` and `--from-key` cannot be set at the same time, choose one."))
	}

	opts := []clientv3.OpOption{}
	switch getConsistency {
	case "s":
		opts = append(opts, clientv3.WithSerializable())
	case "l":
	default:
		ExitWithError(ExitBadFeature, fmt.Errorf("unknown consistency flag %q", getConsistency))
	}

	key := args[0]
	if len(args) > 1 {
		if getPrefix || getFromKey {
			ExitWithError(ExitBadArgs, fmt.Errorf("too many arguments, only accept one argument when `--prefix` or `--from-key` is set."))
		}
		opts = append(opts, clientv3.WithRange(args[1]))
	}

	opts = append(opts, clientv3.WithLimit(getLimit))
	if getRev > 0 {
		opts = append(opts, clientv3.WithRev(getRev))
	}

	sortByOrder := clientv3.SortNone
	sortOrder := strings.ToUpper(getSortOrder)
	switch {
	case sortOrder == "ASCEND":
		sortByOrder = clientv3.SortAscend
	case sortOrder == "DESCEND":
		sortByOrder = clientv3.SortDescend
	case sortOrder == "":
		// nothing
	default:
		ExitWithError(ExitBadFeature, fmt.Errorf("bad sort order %v", getSortOrder))
	}

	sortByTarget := clientv3.SortByKey
	sortTarget := strings.ToUpper(getSortTarget)
	switch {
	case sortTarget == "CREATE":
		sortByTarget = clientv3.SortByCreateRevision
	case sortTarget == "KEY":
		sortByTarget = clientv3.SortByKey
	case sortTarget == "MODIFY":
		sortByTarget = clientv3.SortByModRevision
	case sortTarget == "VALUE":
		sortByTarget = clientv3.SortByValue
	case sortTarget == "VERSION":
		sortByTarget = clientv3.SortByVersion
	case sortTarget == "":
		// nothing
	default:
		ExitWithError(ExitBadFeature, fmt.Errorf("bad sort target %v", getSortTarget))
	}

	opts = append(opts, clientv3.WithSort(sortByTarget, sortByOrder))

	if getPrefix {
		if len(key) == 0 {
			key = "\x00"
			opts = append(opts, clientv3.WithFromKey())
		} else {
			opts = append(opts, clientv3.WithPrefix())
		}
	}

	if getFromKey {
		if len(key) == 0 {
			key = "\x00"
		}
		opts = append(opts, clientv3.WithFromKey())
	}

	if getKeysOnly {
		opts = append(opts, clientv3.WithKeysOnly())
	}

	return key, opts
}
