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
	"time"

	"github.com/coreos/etcd/client"
	"github.com/urfave/cli"
)

// NewUpdateDirCommand returns the CLI command for "updatedir".
func NewUpdateDirCommand() cli.Command {
	return cli.Command{
		Name:      "updatedir",
		Usage:     "update an existing directory",
		ArgsUsage: "<key> <value>",
		Flags: []cli.Flag{
			cli.IntFlag{Name: "ttl", Value: 0, Usage: "key time-to-live in seconds"},
		},
		Action: func(c *cli.Context) error {
			updatedirCommandFunc(c, mustNewKeyAPI(c))
			return nil
		},
	}
}

// updatedirCommandFunc executes the "updatedir" command.
func updatedirCommandFunc(c *cli.Context, ki client.KeysAPI) {
	if len(c.Args()) == 0 {
		handleError(c, ExitBadArgs, errors.New("key required"))
	}
	key := c.Args()[0]
	ttl := c.Int("ttl")
	ctx, cancel := contextWithTotalTimeout(c)
	resp, err := ki.Set(ctx, key, "", &client.SetOptions{TTL: time.Duration(ttl) * time.Second, Dir: true, PrevExist: client.PrevExist})
	cancel()
	if err != nil {
		handleError(c, ExitServerError, err)
	}
	if c.GlobalString("output") != "simple" {
		printResponseKey(resp, c.GlobalString("output"))
	}
}
