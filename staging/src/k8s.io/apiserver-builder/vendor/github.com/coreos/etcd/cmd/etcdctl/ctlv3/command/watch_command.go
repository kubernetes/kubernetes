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
	"bufio"
	"fmt"
	"os"
	"strings"

	"github.com/coreos/etcd/clientv3"
	"github.com/spf13/cobra"
	"golang.org/x/net/context"
)

var (
	watchRev         int64
	watchPrefix      bool
	watchInteractive bool
	watchPrevKey     bool
)

// NewWatchCommand returns the cobra command for "watch".
func NewWatchCommand() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "watch [options] [key or prefix] [range_end]",
		Short: "Watches events stream on keys or prefixes",
		Run:   watchCommandFunc,
	}

	cmd.Flags().BoolVarP(&watchInteractive, "interactive", "i", false, "Interactive mode")
	cmd.Flags().BoolVar(&watchPrefix, "prefix", false, "Watch on a prefix if prefix is set")
	cmd.Flags().Int64Var(&watchRev, "rev", 0, "Revision to start watching")
	cmd.Flags().BoolVar(&watchPrevKey, "prev-kv", false, "get the previous key-value pair before the event happens")

	return cmd
}

// watchCommandFunc executes the "watch" command.
func watchCommandFunc(cmd *cobra.Command, args []string) {
	if watchInteractive {
		watchInteractiveFunc(cmd, args)
		return
	}

	c := mustClientFromCmd(cmd)
	wc, err := getWatchChan(c, args)
	if err != nil {
		ExitWithError(ExitBadArgs, err)
	}

	printWatchCh(wc)
	if err = c.Close(); err != nil {
		ExitWithError(ExitBadConnection, err)
	}
	ExitWithError(ExitInterrupted, fmt.Errorf("watch is canceled by the server"))
}

func watchInteractiveFunc(cmd *cobra.Command, args []string) {
	c := mustClientFromCmd(cmd)

	reader := bufio.NewReader(os.Stdin)

	for {
		l, err := reader.ReadString('\n')
		if err != nil {
			ExitWithError(ExitInvalidInput, fmt.Errorf("Error reading watch request line: %v", err))
		}
		l = strings.TrimSuffix(l, "\n")

		args := argify(l)
		if len(args) < 2 {
			fmt.Fprintf(os.Stderr, "Invalid command %s (command type or key is not provided)\n", l)
			continue
		}

		if args[0] != "watch" {
			fmt.Fprintf(os.Stderr, "Invalid command %s (only support watch)\n", l)
			continue
		}

		flagset := NewWatchCommand().Flags()
		err = flagset.Parse(args[1:])
		if err != nil {
			fmt.Fprintf(os.Stderr, "Invalid command %s (%v)\n", l, err)
			continue
		}
		ch, err := getWatchChan(c, flagset.Args())
		if err != nil {
			fmt.Fprintf(os.Stderr, "Invalid command %s (%v)\n", l, err)
			continue
		}
		go printWatchCh(ch)
	}
}

func getWatchChan(c *clientv3.Client, args []string) (clientv3.WatchChan, error) {
	if len(args) < 1 || len(args) > 2 {
		return nil, fmt.Errorf("bad number of arguments")
	}
	key := args[0]
	opts := []clientv3.OpOption{clientv3.WithRev(watchRev)}
	if len(args) == 2 {
		if watchPrefix {
			return nil, fmt.Errorf("`range_end` and `--prefix` are mutually exclusive")
		}
		opts = append(opts, clientv3.WithRange(args[1]))
	}
	if watchPrefix {
		opts = append(opts, clientv3.WithPrefix())
	}
	if watchPrevKey {
		opts = append(opts, clientv3.WithPrevKV())
	}
	return c.Watch(context.TODO(), key, opts...), nil
}

func printWatchCh(ch clientv3.WatchChan) {
	for resp := range ch {
		display.Watch(resp)
	}
}
