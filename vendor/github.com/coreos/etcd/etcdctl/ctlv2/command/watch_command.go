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
	"errors"
	"fmt"
	"os"
	"os/signal"

	"github.com/coreos/etcd/client"
	"github.com/urfave/cli"
	"golang.org/x/net/context"
)

// NewWatchCommand returns the CLI command for "watch".
func NewWatchCommand() cli.Command {
	return cli.Command{
		Name:      "watch",
		Usage:     "watch a key for changes",
		ArgsUsage: "<key>",
		Flags: []cli.Flag{
			cli.BoolFlag{Name: "forever, f", Usage: "forever watch a key until CTRL+C"},
			cli.IntFlag{Name: "after-index", Value: 0, Usage: "watch after the given index"},
			cli.BoolFlag{Name: "recursive, r", Usage: "returns all values for key and child keys"},
		},
		Action: func(c *cli.Context) error {
			watchCommandFunc(c, mustNewKeyAPI(c))
			return nil
		},
	}
}

// watchCommandFunc executes the "watch" command.
func watchCommandFunc(c *cli.Context, ki client.KeysAPI) {
	if len(c.Args()) == 0 {
		handleError(c, ExitBadArgs, errors.New("key required"))
	}
	key := c.Args()[0]
	recursive := c.Bool("recursive")
	forever := c.Bool("forever")
	index := c.Int("after-index")

	stop := false
	w := ki.Watcher(key, &client.WatcherOptions{AfterIndex: uint64(index), Recursive: recursive})

	sigch := make(chan os.Signal, 1)
	signal.Notify(sigch, os.Interrupt)

	go func() {
		<-sigch
		os.Exit(0)
	}()

	for !stop {
		resp, err := w.Next(context.TODO())
		if err != nil {
			handleError(c, ExitServerError, err)
		}
		if resp.Node.Dir {
			continue
		}
		if recursive {
			fmt.Printf("[%s] %s\n", resp.Action, resp.Node.Key)
		}

		printResponseKey(resp, c.GlobalString("output"))

		if !forever {
			stop = true
		}
	}
}
