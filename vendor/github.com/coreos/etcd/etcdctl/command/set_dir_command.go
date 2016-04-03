// Copyright 2015 CoreOS, Inc.
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
	"github.com/codegangsta/cli"
	"github.com/coreos/etcd/client"
)

// NewSetDirCommand returns the CLI command for "setDir".
func NewSetDirCommand() cli.Command {
	return cli.Command{
		Name:      "setdir",
		Usage:     "create a new directory or update an existing directory TTL",
		ArgsUsage: "<key>",
		Flags: []cli.Flag{
			cli.IntFlag{Name: "ttl", Value: 0, Usage: "key time-to-live"},
		},
		Action: func(c *cli.Context) {
			mkdirCommandFunc(c, mustNewKeyAPI(c), client.PrevIgnore)
		},
	}
}
