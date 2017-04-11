// Copyright 2016 The etcd Authors
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
	"strings"

	"github.com/bgentry/speakeasy"
	"github.com/spf13/cobra"
	"golang.org/x/net/context"
)

var (
	userShowDetail bool
)

// NewUserCommand returns the cobra command for "user".
func NewUserCommand() *cobra.Command {
	ac := &cobra.Command{
		Use:   "user <subcommand>",
		Short: "User related commands",
	}

	ac.AddCommand(newUserAddCommand())
	ac.AddCommand(newUserDeleteCommand())
	ac.AddCommand(newUserGetCommand())
	ac.AddCommand(newUserListCommand())
	ac.AddCommand(newUserChangePasswordCommand())
	ac.AddCommand(newUserGrantRoleCommand())
	ac.AddCommand(newUserRevokeRoleCommand())

	return ac
}

var (
	passwordInteractive bool
)

func newUserAddCommand() *cobra.Command {
	cmd := cobra.Command{
		Use:   "add <user name>",
		Short: "Adds a new user",
		Run:   userAddCommandFunc,
	}

	cmd.Flags().BoolVar(&passwordInteractive, "interactive", true, "Read password from stdin instead of interactive terminal")

	return &cmd
}

func newUserDeleteCommand() *cobra.Command {
	return &cobra.Command{
		Use:   "delete <user name>",
		Short: "Deletes a user",
		Run:   userDeleteCommandFunc,
	}
}

func newUserGetCommand() *cobra.Command {
	cmd := cobra.Command{
		Use:   "get <user name>",
		Short: "Gets detailed information of a user",
		Run:   userGetCommandFunc,
	}

	cmd.Flags().BoolVar(&userShowDetail, "detail", false, "Show permissions of roles granted to the user")

	return &cmd
}

func newUserListCommand() *cobra.Command {
	return &cobra.Command{
		Use:   "list",
		Short: "Lists all users",
		Run:   userListCommandFunc,
	}
}

func newUserChangePasswordCommand() *cobra.Command {
	cmd := cobra.Command{
		Use:   "passwd <user name>",
		Short: "Changes password of user",
		Run:   userChangePasswordCommandFunc,
	}

	cmd.Flags().BoolVar(&passwordInteractive, "interactive", true, "If true, read password from stdin instead of interactive terminal")

	return &cmd
}

func newUserGrantRoleCommand() *cobra.Command {
	return &cobra.Command{
		Use:   "grant-role <user name> <role name>",
		Short: "Grants a role to a user",
		Run:   userGrantRoleCommandFunc,
	}
}

func newUserRevokeRoleCommand() *cobra.Command {
	return &cobra.Command{
		Use:   "revoke-role <user name> <role name>",
		Short: "Revokes a role from a user",
		Run:   userRevokeRoleCommandFunc,
	}
}

// userAddCommandFunc executes the "user add" command.
func userAddCommandFunc(cmd *cobra.Command, args []string) {
	if len(args) != 1 {
		ExitWithError(ExitBadArgs, fmt.Errorf("user add command requires user name as its argument."))
	}

	var password string

	if !passwordInteractive {
		fmt.Scanf("%s", &password)
	} else {
		password = readPasswordInteractive(args[0])
	}

	_, err := mustClientFromCmd(cmd).Auth.UserAdd(context.TODO(), args[0], password)
	if err != nil {
		ExitWithError(ExitError, err)
	}

	fmt.Printf("User %s created\n", args[0])
}

// userDeleteCommandFunc executes the "user delete" command.
func userDeleteCommandFunc(cmd *cobra.Command, args []string) {
	if len(args) != 1 {
		ExitWithError(ExitBadArgs, fmt.Errorf("user delete command requires user name as its argument."))
	}

	_, err := mustClientFromCmd(cmd).Auth.UserDelete(context.TODO(), args[0])
	if err != nil {
		ExitWithError(ExitError, err)
	}

	fmt.Printf("User %s deleted\n", args[0])
}

// userGetCommandFunc executes the "user get" command.
func userGetCommandFunc(cmd *cobra.Command, args []string) {
	if len(args) != 1 {
		ExitWithError(ExitBadArgs, fmt.Errorf("user get command requires user name as its argument."))
	}

	name := args[0]
	client := mustClientFromCmd(cmd)
	resp, err := client.Auth.UserGet(context.TODO(), name)
	if err != nil {
		ExitWithError(ExitError, err)
	}

	fmt.Printf("User: %s\n", name)
	if !userShowDetail {
		fmt.Printf("Roles:")
		for _, role := range resp.Roles {
			fmt.Printf(" %s", role)
		}
		fmt.Printf("\n")
	} else {
		for _, role := range resp.Roles {
			fmt.Printf("\n")
			roleResp, err := client.Auth.RoleGet(context.TODO(), role)
			if err != nil {
				ExitWithError(ExitError, err)
			}

			printRolePermissions(role, roleResp)
		}
	}
}

// userListCommandFunc executes the "user list" command.
func userListCommandFunc(cmd *cobra.Command, args []string) {
	if len(args) != 0 {
		ExitWithError(ExitBadArgs, fmt.Errorf("user list command requires no arguments."))
	}

	resp, err := mustClientFromCmd(cmd).Auth.UserList(context.TODO())
	if err != nil {
		ExitWithError(ExitError, err)
	}

	for _, user := range resp.Users {
		fmt.Printf("%s\n", user)
	}
}

// userChangePasswordCommandFunc executes the "user passwd" command.
func userChangePasswordCommandFunc(cmd *cobra.Command, args []string) {
	if len(args) != 1 {
		ExitWithError(ExitBadArgs, fmt.Errorf("user passwd command requires user name as its argument."))
	}

	var password string

	if !passwordInteractive {
		fmt.Scanf("%s", &password)
	} else {
		password = readPasswordInteractive(args[0])
	}

	_, err := mustClientFromCmd(cmd).Auth.UserChangePassword(context.TODO(), args[0], password)
	if err != nil {
		ExitWithError(ExitError, err)
	}

	fmt.Println("Password updated")
}

// userGrantRoleCommandFunc executes the "user grant-role" command.
func userGrantRoleCommandFunc(cmd *cobra.Command, args []string) {
	if len(args) != 2 {
		ExitWithError(ExitBadArgs, fmt.Errorf("user grant command requires user name and role name as its argument."))
	}

	_, err := mustClientFromCmd(cmd).Auth.UserGrantRole(context.TODO(), args[0], args[1])
	if err != nil {
		ExitWithError(ExitError, err)
	}

	fmt.Printf("Role %s is granted to user %s\n", args[1], args[0])
}

// userRevokeRoleCommandFunc executes the "user revoke-role" command.
func userRevokeRoleCommandFunc(cmd *cobra.Command, args []string) {
	if len(args) != 2 {
		ExitWithError(ExitBadArgs, fmt.Errorf("user revoke-role requires user name and role name as its argument."))
	}

	_, err := mustClientFromCmd(cmd).Auth.UserRevokeRole(context.TODO(), args[0], args[1])
	if err != nil {
		ExitWithError(ExitError, err)
	}

	fmt.Printf("Role %s is revoked from user %s\n", args[1], args[0])
}

func readPasswordInteractive(name string) string {
	prompt1 := fmt.Sprintf("Password of %s: ", name)
	password1, err1 := speakeasy.Ask(prompt1)
	if err1 != nil {
		ExitWithError(ExitBadArgs, fmt.Errorf("failed to ask password: %s.", err1))
	}

	if len(password1) == 0 {
		ExitWithError(ExitBadArgs, fmt.Errorf("empty password"))
	}

	prompt2 := fmt.Sprintf("Type password of %s again for confirmation: ", name)
	password2, err2 := speakeasy.Ask(prompt2)
	if err2 != nil {
		ExitWithError(ExitBadArgs, fmt.Errorf("failed to ask password: %s.", err2))
	}

	if strings.Compare(password1, password2) != 0 {
		ExitWithError(ExitBadArgs, fmt.Errorf("given passwords are different."))
	}

	return password1
}
