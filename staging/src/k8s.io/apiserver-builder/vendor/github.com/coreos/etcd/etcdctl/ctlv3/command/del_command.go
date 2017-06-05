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

	"github.com/coreos/etcd/clientv3"
	"github.com/spf13/cobra"
)

var (
	delPrefix bool
	delPrevKV bool
)

// NewDelCommand returns the cobra command for "del".
func NewDelCommand() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "del [options] <key> [range_end]",
		Short: "Removes the specified key or range of keys [key, range_end)",
		Run:   delCommandFunc,
	}

	cmd.Flags().BoolVar(&delPrefix, "prefix", false, "delete keys with matching prefix")
	cmd.Flags().BoolVar(&delPrevKV, "prev-kv", false, "return deleted key-value pairs")
	return cmd
}

// delCommandFunc executes the "del" command.
func delCommandFunc(cmd *cobra.Command, args []string) {
	key, opts := getDelOp(cmd, args)
	ctx, cancel := commandCtx(cmd)
	resp, err := mustClientFromCmd(cmd).Delete(ctx, key, opts...)
	cancel()
	if err != nil {
		ExitWithError(ExitError, err)
	}
	display.Del(*resp)
}

func getDelOp(cmd *cobra.Command, args []string) (string, []clientv3.OpOption) {
	if len(args) == 0 || len(args) > 2 {
		ExitWithError(ExitBadArgs, fmt.Errorf("del command needs one argument as key and an optional argument as range_end."))
	}
	opts := []clientv3.OpOption{}
	key := args[0]
	if len(args) > 1 {
		if delPrefix {
			ExitWithError(ExitBadArgs, fmt.Errorf("too many arguments, only accept one arguement when `--prefix` is set."))
		}
		opts = append(opts, clientv3.WithRange(args[1]))
	}

	if delPrefix {
		opts = append(opts, clientv3.WithPrefix())
	}
	if delPrevKV {
		opts = append(opts, clientv3.WithPrevKV())
	}

	return key, opts
}
