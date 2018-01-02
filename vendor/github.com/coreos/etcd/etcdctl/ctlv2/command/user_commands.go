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

	"github.com/bgentry/speakeasy"
	"github.com/coreos/etcd/client"
	"github.com/urfave/cli"
)

func NewUserCommands() cli.Command {
	return cli.Command{
		Name:  "user",
		Usage: "user add, grant and revoke subcommands",
		Subcommands: []cli.Command{
			{
				Name:      "add",
				Usage:     "add a new user for the etcd cluster",
				ArgsUsage: "<user>",
				Action:    actionUserAdd,
			},
			{
				Name:      "get",
				Usage:     "get details for a user",
				ArgsUsage: "<user>",
				Action:    actionUserGet,
			},
			{
				Name:      "list",
				Usage:     "list all current users",
				ArgsUsage: "<user>",
				Action:    actionUserList,
			},
			{
				Name:      "remove",
				Usage:     "remove a user for the etcd cluster",
				ArgsUsage: "<user>",
				Action:    actionUserRemove,
			},
			{
				Name:      "grant",
				Usage:     "grant roles to an etcd user",
				ArgsUsage: "<user>",
				Flags:     []cli.Flag{cli.StringSliceFlag{Name: "roles", Value: new(cli.StringSlice), Usage: "List of roles to grant or revoke"}},
				Action:    actionUserGrant,
			},
			{
				Name:      "revoke",
				Usage:     "revoke roles for an etcd user",
				ArgsUsage: "<user>",
				Flags:     []cli.Flag{cli.StringSliceFlag{Name: "roles", Value: new(cli.StringSlice), Usage: "List of roles to grant or revoke"}},
				Action:    actionUserRevoke,
			},
			{
				Name:      "passwd",
				Usage:     "change password for a user",
				ArgsUsage: "<user>",
				Action:    actionUserPasswd,
			},
		},
	}
}

func mustNewAuthUserAPI(c *cli.Context) client.AuthUserAPI {
	hc := mustNewClient(c)

	if c.GlobalBool("debug") {
		fmt.Fprintf(os.Stderr, "Cluster-Endpoints: %s\n", strings.Join(hc.Endpoints(), ", "))
	}

	return client.NewAuthUserAPI(hc)
}

func actionUserList(c *cli.Context) error {
	if len(c.Args()) != 0 {
		fmt.Fprintln(os.Stderr, "No arguments accepted")
		os.Exit(1)
	}
	u := mustNewAuthUserAPI(c)
	ctx, cancel := contextWithTotalTimeout(c)
	users, err := u.ListUsers(ctx)
	cancel()
	if err != nil {
		fmt.Fprintln(os.Stderr, err.Error())
		os.Exit(1)
	}

	for _, user := range users {
		fmt.Printf("%s\n", user)
	}
	return nil
}

func actionUserAdd(c *cli.Context) error {
	api, userarg := mustUserAPIAndName(c)
	ctx, cancel := contextWithTotalTimeout(c)
	defer cancel()
	user, _, _ := getUsernamePassword("", userarg+":")

	_, pass, err := getUsernamePassword("New password: ", userarg)
	if err != nil {
		fmt.Fprintln(os.Stderr, "Error reading password:", err)
		os.Exit(1)
	}
	err = api.AddUser(ctx, user, pass)
	if err != nil {
		fmt.Fprintln(os.Stderr, err.Error())
		os.Exit(1)
	}

	fmt.Printf("User %s created\n", user)
	return nil
}

func actionUserRemove(c *cli.Context) error {
	api, user := mustUserAPIAndName(c)
	ctx, cancel := contextWithTotalTimeout(c)
	err := api.RemoveUser(ctx, user)
	cancel()
	if err != nil {
		fmt.Fprintln(os.Stderr, err.Error())
		os.Exit(1)
	}

	fmt.Printf("User %s removed\n", user)
	return nil
}

func actionUserPasswd(c *cli.Context) error {
	api, user := mustUserAPIAndName(c)
	ctx, cancel := contextWithTotalTimeout(c)
	defer cancel()
	pass, err := speakeasy.Ask("New password: ")
	if err != nil {
		fmt.Fprintln(os.Stderr, "Error reading password:", err)
		os.Exit(1)
	}

	_, err = api.ChangePassword(ctx, user, pass)
	if err != nil {
		fmt.Fprintln(os.Stderr, err.Error())
		os.Exit(1)
	}

	fmt.Printf("Password updated\n")
	return nil
}

func actionUserGrant(c *cli.Context) error {
	userGrantRevoke(c, true)
	return nil
}

func actionUserRevoke(c *cli.Context) error {
	userGrantRevoke(c, false)
	return nil
}

func userGrantRevoke(c *cli.Context, grant bool) {
	roles := c.StringSlice("roles")
	if len(roles) == 0 {
		fmt.Fprintln(os.Stderr, "No roles specified; please use `--roles`")
		os.Exit(1)
	}

	ctx, cancel := contextWithTotalTimeout(c)
	defer cancel()

	api, user := mustUserAPIAndName(c)
	var err error
	if grant {
		_, err = api.GrantUser(ctx, user, roles)
	} else {
		_, err = api.RevokeUser(ctx, user, roles)
	}

	if err != nil {
		fmt.Fprintln(os.Stderr, err.Error())
		os.Exit(1)
	}

	fmt.Printf("User %s updated\n", user)
}

func actionUserGet(c *cli.Context) error {
	api, username := mustUserAPIAndName(c)
	ctx, cancel := contextWithTotalTimeout(c)
	user, err := api.GetUser(ctx, username)
	cancel()
	if err != nil {
		fmt.Fprintln(os.Stderr, err.Error())
		os.Exit(1)
	}
	fmt.Printf("User: %s\n", user.User)
	fmt.Printf("Roles: %s\n", strings.Join(user.Roles, " "))
	return nil
}

func mustUserAPIAndName(c *cli.Context) (client.AuthUserAPI, string) {
	args := c.Args()
	if len(args) != 1 {
		fmt.Fprintln(os.Stderr, "Please provide a username")
		os.Exit(1)
	}

	api := mustNewAuthUserAPI(c)
	username := args[0]
	return api, username
}
