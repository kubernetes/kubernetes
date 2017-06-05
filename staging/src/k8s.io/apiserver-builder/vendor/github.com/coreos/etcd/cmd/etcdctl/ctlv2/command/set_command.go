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

// NewSetCommand returns the CLI command for "set".
func NewSetCommand() cli.Command {
	return cli.Command{
		Name:      "set",
		Usage:     "set the value of a key",
		ArgsUsage: "<key> <value>",
		Description: `Set sets the value of a key.

   When <value> begins with '-', <value> is interpreted as a flag.
   Insert '--' for workaround:

   $ set -- <key> <value>`,
		Flags: []cli.Flag{
			cli.IntFlag{Name: "ttl", Value: 0, Usage: "key time-to-live"},
			cli.StringFlag{Name: "swap-with-value", Value: "", Usage: "previous value"},
			cli.IntFlag{Name: "swap-with-index", Value: 0, Usage: "previous index"},
		},
		Action: func(c *cli.Context) error {
			setCommandFunc(c, mustNewKeyAPI(c))
			return nil
		},
	}
}

// setCommandFunc executes the "set" command.
func setCommandFunc(c *cli.Context, ki client.KeysAPI) {
	if len(c.Args()) == 0 {
		handleError(ExitBadArgs, errors.New("key required"))
	}
	key := c.Args()[0]
	value, err := argOrStdin(c.Args(), os.Stdin, 1)
	if err != nil {
		handleError(ExitBadArgs, errors.New("value required"))
	}

	ttl := c.Int("ttl")
	prevValue := c.String("swap-with-value")
	prevIndex := c.Int("swap-with-index")

	ctx, cancel := contextWithTotalTimeout(c)
	resp, err := ki.Set(ctx, key, value, &client.SetOptions{TTL: time.Duration(ttl) * time.Second, PrevIndex: uint64(prevIndex), PrevValue: prevValue})
	cancel()
	if err != nil {
		handleError(ExitServerError, err)
	}

	printResponseKey(resp, c.GlobalString("output"))
}
