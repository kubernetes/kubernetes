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
	"os"
	"strconv"

	"github.com/coreos/etcd/clientv3"
	"github.com/spf13/cobra"
)

var (
	leaseStr  string
	putPrevKV bool
)

// NewPutCommand returns the cobra command for "put".
func NewPutCommand() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "put [options] <key> <value> (<value> can also be given from stdin)",
		Short: "Puts the given key into the store",
		Long: `
Puts the given key into the store.

When <value> begins with '-', <value> is interpreted as a flag.
Insert '--' for workaround:

$ put <key> -- <value>
$ put -- <key> <value>

If <value> isn't given as command line argument, this command tries to read the value from standard input.
For example,
$ cat file | put <key>
will store the content of the file to <key>.
`,
		Run: putCommandFunc,
	}
	cmd.Flags().StringVar(&leaseStr, "lease", "0", "lease ID (in hexadecimal) to attach to the key")
	cmd.Flags().BoolVar(&putPrevKV, "prev-kv", false, "return the previous key-value pair before modification")
	return cmd
}

// putCommandFunc executes the "put" command.
func putCommandFunc(cmd *cobra.Command, args []string) {
	key, value, opts := getPutOp(cmd, args)

	ctx, cancel := commandCtx(cmd)
	resp, err := mustClientFromCmd(cmd).Put(ctx, key, value, opts...)
	cancel()
	if err != nil {
		ExitWithError(ExitError, err)
	}
	display.Put(*resp)
}

func getPutOp(cmd *cobra.Command, args []string) (string, string, []clientv3.OpOption) {
	if len(args) == 0 {
		ExitWithError(ExitBadArgs, fmt.Errorf("put command needs 1 argument and input from stdin or 2 arguments."))
	}

	key := args[0]
	value, err := argOrStdin(args, os.Stdin, 1)
	if err != nil {
		ExitWithError(ExitBadArgs, fmt.Errorf("put command needs 1 argument and input from stdin or 2 arguments."))
	}

	id, err := strconv.ParseInt(leaseStr, 16, 64)
	if err != nil {
		ExitWithError(ExitBadArgs, fmt.Errorf("bad lease ID (%v), expecting ID in Hex", err))
	}

	opts := []clientv3.OpOption{}
	if id != 0 {
		opts = append(opts, clientv3.WithLease(clientv3.LeaseID(id)))
	}
	if putPrevKV {
		opts = append(opts, clientv3.WithPrevKV())
	}

	return key, value, opts
}
