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
	"os"
	"strings"

	"github.com/coreos/etcd/client"
	"github.com/urfave/cli"
)

func NewAuthCommands() cli.Command {
	return cli.Command{
		Name:  "auth",
		Usage: "overall auth controls",
		Subcommands: []cli.Command{
			{
				Name:      "enable",
				Usage:     "enable auth access controls",
				ArgsUsage: " ",
				Action:    actionAuthEnable,
			},
			{
				Name:      "disable",
				Usage:     "disable auth access controls",
				ArgsUsage: " ",
				Action:    actionAuthDisable,
			},
		},
	}
}

func actionAuthEnable(c *cli.Context) error {
	authEnableDisable(c, true)
	return nil
}

func actionAuthDisable(c *cli.Context) error {
	authEnableDisable(c, false)
	return nil
}

func mustNewAuthAPI(c *cli.Context) client.AuthAPI {
	hc := mustNewClient(c)

	if c.GlobalBool("debug") {
		fmt.Fprintf(os.Stderr, "Cluster-Endpoints: %s\n", strings.Join(hc.Endpoints(), ", "))
	}

	return client.NewAuthAPI(hc)
}

func authEnableDisable(c *cli.Context, enable bool) {
	if len(c.Args()) != 0 {
		fmt.Fprintln(os.Stderr, "No arguments accepted")
		os.Exit(1)
	}
	s := mustNewAuthAPI(c)
	ctx, cancel := contextWithTotalTimeout(c)
	var err error
	if enable {
		err = s.Enable(ctx)
	} else {
		err = s.Disable(ctx)
	}
	cancel()
	if err != nil {
		fmt.Fprintln(os.Stderr, err.Error())
		os.Exit(1)
	}
	if enable {
		fmt.Println("Authentication Enabled")
	} else {
		fmt.Println("Authentication Disabled")
	}
}
