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

	"github.com/coreos/etcd/client"
	"github.com/urfave/cli"
)

// NewGetCommand returns the CLI command for "get".
func NewGetCommand() cli.Command {
	return cli.Command{
		Name:      "get",
		Usage:     "retrieve the value of a key",
		ArgsUsage: "<key>",
		Flags: []cli.Flag{
			cli.BoolFlag{Name: "sort", Usage: "returns result in sorted order"},
			cli.BoolFlag{Name: "quorum, q", Usage: "require quorum for get request"},
		},
		Action: func(c *cli.Context) error {
			getCommandFunc(c, mustNewKeyAPI(c))
			return nil
		},
	}
}

// getCommandFunc executes the "get" command.
func getCommandFunc(c *cli.Context, ki client.KeysAPI) {
	if len(c.Args()) == 0 {
		handleError(c, ExitBadArgs, errors.New("key required"))
	}

	key := c.Args()[0]
	sorted := c.Bool("sort")
	quorum := c.Bool("quorum")

	ctx, cancel := contextWithTotalTimeout(c)
	resp, err := ki.Get(ctx, key, &client.GetOptions{Sort: sorted, Quorum: quorum})
	cancel()
	if err != nil {
		handleError(c, ExitServerError, err)
	}

	if resp.Node.Dir {
		fmt.Fprintln(os.Stderr, fmt.Sprintf("%s: is a directory", resp.Node.Key))
		os.Exit(1)
	}

	printResponseKey(resp, c.GlobalString("output"))
}
