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
	"os"
	"time"

	"github.com/coreos/etcd/client"
	"github.com/urfave/cli"
)

// NewMakeCommand returns the CLI command for "mk".
func NewMakeCommand() cli.Command {
	return cli.Command{
		Name:      "mk",
		Usage:     "make a new key with a given value",
		ArgsUsage: "<key> <value>",
		Flags: []cli.Flag{
			cli.BoolFlag{Name: "in-order", Usage: "create in-order key under directory <key>"},
			cli.IntFlag{Name: "ttl", Value: 0, Usage: "key time-to-live in seconds"},
		},
		Action: func(c *cli.Context) error {
			mkCommandFunc(c, mustNewKeyAPI(c))
			return nil
		},
	}
}

// mkCommandFunc executes the "mk" command.
func mkCommandFunc(c *cli.Context, ki client.KeysAPI) {
	if len(c.Args()) == 0 {
		handleError(c, ExitBadArgs, errors.New("key required"))
	}
	key := c.Args()[0]
	value, err := argOrStdin(c.Args(), os.Stdin, 1)
	if err != nil {
		handleError(c, ExitBadArgs, errors.New("value required"))
	}

	ttl := c.Int("ttl")
	inorder := c.Bool("in-order")

	var resp *client.Response
	ctx, cancel := contextWithTotalTimeout(c)
	if !inorder {
		// Since PrevNoExist means that the Node must not exist previously,
		// this Set method always creates a new key. Therefore, mk command
		// succeeds only if the key did not previously exist, and the command
		// prevents one from overwriting values accidentally.
		resp, err = ki.Set(ctx, key, value, &client.SetOptions{TTL: time.Duration(ttl) * time.Second, PrevExist: client.PrevNoExist})
	} else {
		// If in-order flag is specified then create an inorder key under
		// the directory identified by the key argument.
		resp, err = ki.CreateInOrder(ctx, key, value, &client.CreateInOrderOptions{TTL: time.Duration(ttl) * time.Second})
	}
	cancel()
	if err != nil {
		handleError(c, ExitServerError, err)
	}

	printResponseKey(resp, c.GlobalString("output"))
}
