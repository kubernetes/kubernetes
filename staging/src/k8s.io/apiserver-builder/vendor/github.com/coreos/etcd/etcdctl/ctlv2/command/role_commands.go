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
	"reflect"
	"strings"

	"github.com/coreos/etcd/client"
	"github.com/coreos/etcd/pkg/pathutil"
	"github.com/urfave/cli"
)

func NewRoleCommands() cli.Command {
	return cli.Command{
		Name:  "role",
		Usage: "role add, grant and revoke subcommands",
		Subcommands: []cli.Command{
			{
				Name:      "add",
				Usage:     "add a new role for the etcd cluster",
				ArgsUsage: "<role> ",
				Action:    actionRoleAdd,
			},
			{
				Name:      "get",
				Usage:     "get details for a role",
				ArgsUsage: "<role>",
				Action:    actionRoleGet,
			},
			{
				Name:      "list",
				Usage:     "list all roles",
				ArgsUsage: " ",
				Action:    actionRoleList,
			},
			{
				Name:      "remove",
				Usage:     "remove a role from the etcd cluster",
				ArgsUsage: "<role>",
				Action:    actionRoleRemove,
			},
			{
				Name:      "grant",
				Usage:     "grant path matches to an etcd role",
				ArgsUsage: "<role>",
				Flags: []cli.Flag{
					cli.StringFlag{Name: "path", Value: "", Usage: "Path granted for the role to access"},
					cli.BoolFlag{Name: "read", Usage: "Grant read-only access"},
					cli.BoolFlag{Name: "write", Usage: "Grant write-only access"},
					cli.BoolFlag{Name: "readwrite, rw", Usage: "Grant read-write access"},
				},
				Action: actionRoleGrant,
			},
			{
				Name:      "revoke",
				Usage:     "revoke path matches for an etcd role",
				ArgsUsage: "<role>",
				Flags: []cli.Flag{
					cli.StringFlag{Name: "path", Value: "", Usage: "Path revoked for the role to access"},
					cli.BoolFlag{Name: "read", Usage: "Revoke read access"},
					cli.BoolFlag{Name: "write", Usage: "Revoke write access"},
					cli.BoolFlag{Name: "readwrite, rw", Usage: "Revoke read-write access"},
				},
				Action: actionRoleRevoke,
			},
		},
	}
}

func mustNewAuthRoleAPI(c *cli.Context) client.AuthRoleAPI {
	hc := mustNewClient(c)

	if c.GlobalBool("debug") {
		fmt.Fprintf(os.Stderr, "Cluster-Endpoints: %s\n", strings.Join(hc.Endpoints(), ", "))
	}

	return client.NewAuthRoleAPI(hc)
}

func actionRoleList(c *cli.Context) error {
	if len(c.Args()) != 0 {
		fmt.Fprintln(os.Stderr, "No arguments accepted")
		os.Exit(1)
	}
	r := mustNewAuthRoleAPI(c)
	ctx, cancel := contextWithTotalTimeout(c)
	roles, err := r.ListRoles(ctx)
	cancel()
	if err != nil {
		fmt.Fprintln(os.Stderr, err.Error())
		os.Exit(1)
	}

	for _, role := range roles {
		fmt.Printf("%s\n", role)
	}

	return nil
}

func actionRoleAdd(c *cli.Context) error {
	api, role := mustRoleAPIAndName(c)
	ctx, cancel := contextWithTotalTimeout(c)
	defer cancel()
	currentRole, err := api.GetRole(ctx, role)
	if currentRole != nil {
		fmt.Fprintf(os.Stderr, "Role %s already exists\n", role)
		os.Exit(1)
	}

	err = api.AddRole(ctx, role)
	if err != nil {
		fmt.Fprintln(os.Stderr, err.Error())
		os.Exit(1)
	}

	fmt.Printf("Role %s created\n", role)
	return nil
}

func actionRoleRemove(c *cli.Context) error {
	api, role := mustRoleAPIAndName(c)
	ctx, cancel := contextWithTotalTimeout(c)
	err := api.RemoveRole(ctx, role)
	cancel()
	if err != nil {
		fmt.Fprintln(os.Stderr, err.Error())
		os.Exit(1)
	}

	fmt.Printf("Role %s removed\n", role)
	return nil
}

func actionRoleGrant(c *cli.Context) error {
	roleGrantRevoke(c, true)
	return nil
}

func actionRoleRevoke(c *cli.Context) error {
	roleGrantRevoke(c, false)
	return nil
}

func roleGrantRevoke(c *cli.Context, grant bool) {
	path := c.String("path")
	if path == "" {
		fmt.Fprintln(os.Stderr, "No path specified; please use `--path`")
		os.Exit(1)
	}
	if pathutil.CanonicalURLPath(path) != path {
		fmt.Fprintf(os.Stderr, "Not canonical path; please use `--path=%s`\n", pathutil.CanonicalURLPath(path))
		os.Exit(1)
	}

	read := c.Bool("read")
	write := c.Bool("write")
	rw := c.Bool("readwrite")
	permcount := 0
	for _, v := range []bool{read, write, rw} {
		if v {
			permcount++
		}
	}
	if permcount != 1 {
		fmt.Fprintln(os.Stderr, "Please specify exactly one of --read, --write or --readwrite")
		os.Exit(1)
	}
	var permType client.PermissionType
	switch {
	case read:
		permType = client.ReadPermission
	case write:
		permType = client.WritePermission
	case rw:
		permType = client.ReadWritePermission
	}

	api, role := mustRoleAPIAndName(c)
	ctx, cancel := contextWithTotalTimeout(c)
	defer cancel()
	currentRole, err := api.GetRole(ctx, role)
	if err != nil {
		fmt.Fprintln(os.Stderr, err.Error())
		os.Exit(1)
	}
	var newRole *client.Role
	if grant {
		newRole, err = api.GrantRoleKV(ctx, role, []string{path}, permType)
	} else {
		newRole, err = api.RevokeRoleKV(ctx, role, []string{path}, permType)
	}
	if err != nil {
		fmt.Fprintln(os.Stderr, err.Error())
		os.Exit(1)
	}
	if reflect.DeepEqual(newRole, currentRole) {
		if grant {
			fmt.Printf("Role unchanged; already granted")
		} else {
			fmt.Printf("Role unchanged; already revoked")
		}
	}

	fmt.Printf("Role %s updated\n", role)
}

func actionRoleGet(c *cli.Context) error {
	api, rolename := mustRoleAPIAndName(c)

	ctx, cancel := contextWithTotalTimeout(c)
	role, err := api.GetRole(ctx, rolename)
	cancel()
	if err != nil {
		fmt.Fprintln(os.Stderr, err.Error())
		os.Exit(1)
	}
	fmt.Printf("Role: %s\n", role.Role)
	fmt.Printf("KV Read:\n")
	for _, v := range role.Permissions.KV.Read {
		fmt.Printf("\t%s\n", v)
	}
	fmt.Printf("KV Write:\n")
	for _, v := range role.Permissions.KV.Write {
		fmt.Printf("\t%s\n", v)
	}
	return nil
}

func mustRoleAPIAndName(c *cli.Context) (client.AuthRoleAPI, string) {
	args := c.Args()
	if len(args) != 1 {
		fmt.Fprintln(os.Stderr, "Please provide a role name")
		os.Exit(1)
	}

	name := args[0]
	api := mustNewAuthRoleAPI(c)
	return api, name
}
