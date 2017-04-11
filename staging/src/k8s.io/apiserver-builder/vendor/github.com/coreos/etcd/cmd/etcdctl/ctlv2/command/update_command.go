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

// NewUpdateCommand returns the CLI command for "update".
func NewUpdateCommand() cli.Command {
	return cli.Command{
		Name:      "update",
		Usage:     "update an existing key with a given value",
		ArgsUsage: "<key> <value>",
		Flags: []cli.Flag{
			cli.IntFlag{Name: "ttl", Value: 0, Usage: "key time-to-live"},
		},
		Action: func(c *cli.Context) error {
			updateCommandFunc(c, mustNewKeyAPI(c))
			return nil
		},
	}
}

// updateCommandFunc executes the "update" command.
func updateCommandFunc(c *cli.Context, ki client.KeysAPI) {
	if len(c.Args()) == 0 {
		handleError(ExitBadArgs, errors.New("key required"))
	}
	key := c.Args()[0]
	value, err := argOrStdin(c.Args(), os.Stdin, 1)
	if err != nil {
		handleError(ExitBadArgs, errors.New("value required"))
	}

	ttl := c.Int("ttl")

	ctx, cancel := contextWithTotalTimeout(c)
	resp, err := ki.Set(ctx, key, value, &client.SetOptions{TTL: time.Duration(ttl) * time.Second, PrevExist: client.PrevExist})
	cancel()
	if err != nil {
		handleError(ExitServerError, err)
	}

	printResponseKey(resp, c.GlobalString("output"))
}
