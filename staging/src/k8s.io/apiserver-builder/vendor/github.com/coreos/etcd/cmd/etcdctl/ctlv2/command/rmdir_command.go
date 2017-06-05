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

	"github.com/coreos/etcd/client"
	"github.com/urfave/cli"
)

// NewRemoveDirCommand returns the CLI command for "rmdir".
func NewRemoveDirCommand() cli.Command {
	return cli.Command{
		Name:      "rmdir",
		Usage:     "removes the key if it is an empty directory or a key-value pair",
		ArgsUsage: "<key>",
		Action: func(c *cli.Context) error {
			rmdirCommandFunc(c, mustNewKeyAPI(c))
			return nil
		},
	}
}

// rmdirCommandFunc executes the "rmdir" command.
func rmdirCommandFunc(c *cli.Context, ki client.KeysAPI) {
	if len(c.Args()) == 0 {
		handleError(ExitBadArgs, errors.New("key required"))
	}
	key := c.Args()[0]

	ctx, cancel := contextWithTotalTimeout(c)
	resp, err := ki.Delete(ctx, key, &client.DeleteOptions{Dir: true})
	cancel()
	if err != nil {
		handleError(ExitServerError, err)
	}

	if !resp.Node.Dir {
		printResponseKey(resp, c.GlobalString("output"))
	}
}
