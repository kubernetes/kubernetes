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

	"github.com/coreos/etcd/client"
	"github.com/urfave/cli"
)

func NewLsCommand() cli.Command {
	return cli.Command{
		Name:      "ls",
		Usage:     "retrieve a directory",
		ArgsUsage: "[key]",
		Flags: []cli.Flag{
			cli.BoolFlag{Name: "sort", Usage: "returns result in sorted order"},
			cli.BoolFlag{Name: "recursive, r", Usage: "returns all key names recursively for the given path"},
			cli.BoolFlag{Name: "p", Usage: "append slash (/) to directories"},
			cli.BoolFlag{Name: "quorum, q", Usage: "require quorum for get request"},
		},
		Action: func(c *cli.Context) error {
			lsCommandFunc(c, mustNewKeyAPI(c))
			return nil
		},
	}
}

// lsCommandFunc executes the "ls" command.
func lsCommandFunc(c *cli.Context, ki client.KeysAPI) {
	key := "/"
	if len(c.Args()) != 0 {
		key = c.Args()[0]
	}

	sort := c.Bool("sort")
	recursive := c.Bool("recursive")
	quorum := c.Bool("quorum")

	ctx, cancel := contextWithTotalTimeout(c)
	resp, err := ki.Get(ctx, key, &client.GetOptions{Sort: sort, Recursive: recursive, Quorum: quorum})
	cancel()
	if err != nil {
		handleError(c, ExitServerError, err)
	}

	printLs(c, resp)
}

// printLs writes a response out in a manner similar to the `ls` command in unix.
// Non-empty directories list their contents and files list their name.
func printLs(c *cli.Context, resp *client.Response) {
	if c.GlobalString("output") == "simple" {
		if !resp.Node.Dir {
			fmt.Println(resp.Node.Key)
		}
		for _, node := range resp.Node.Nodes {
			rPrint(c, node)
		}
	} else {
		// user wants JSON or extended output
		printResponseKey(resp, c.GlobalString("output"))
	}
}

// rPrint recursively prints out the nodes in the node structure.
func rPrint(c *cli.Context, n *client.Node) {
	if n.Dir && c.Bool("p") {
		fmt.Println(fmt.Sprintf("%v/", n.Key))
	} else {
		fmt.Println(n.Key)
	}

	for _, node := range n.Nodes {
		rPrint(c, node)
	}
}
