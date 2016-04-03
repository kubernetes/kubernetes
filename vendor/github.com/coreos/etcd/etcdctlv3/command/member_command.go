// Copyright 2016 CoreOS, Inc.
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
	"strconv"
	"strings"

	"github.com/spf13/cobra"
	"golang.org/x/net/context"
)

var (
	memberID       uint64
	memberPeerURLs string
)

// NewMemberCommand returns the cobra command for "member".
func NewMemberCommand() *cobra.Command {
	mc := &cobra.Command{
		Use:   "member",
		Short: "member is used to manage membership in an etcd cluster.",
	}

	mc.AddCommand(NewMemberAddCommand())
	mc.AddCommand(NewMemberRemoveCommand())
	mc.AddCommand(NewMemberUpdateCommand())
	mc.AddCommand(NewMemberListCommand())

	return mc
}

// NewMemberAddCommand returns the cobra command for "member add".
func NewMemberAddCommand() *cobra.Command {
	cc := &cobra.Command{
		Use:   "add",
		Short: "add is used to add a member into the cluster",

		Run: memberAddCommandFunc,
	}

	cc.Flags().StringVar(&memberPeerURLs, "peerURLs", "", "comma separated peer URLs for the new member.")

	return cc
}

// NewMemberRemoveCommand returns the cobra command for "member remove".
func NewMemberRemoveCommand() *cobra.Command {
	cc := &cobra.Command{
		Use:   "remove",
		Short: "remove is used to remove a member from the cluster",

		Run: memberRemoveCommandFunc,
	}

	return cc
}

// NewMemberUpdateCommand returns the cobra command for "member update".
func NewMemberUpdateCommand() *cobra.Command {
	cc := &cobra.Command{
		Use:   "update",
		Short: "update is used to update a member in the cluster",

		Run: memberUpdateCommandFunc,
	}

	cc.Flags().StringVar(&memberPeerURLs, "peerURLs", "", "comma separated peer URLs for the updated member.")

	return cc
}

// NewMemberListCommand returns the cobra command for "member list".
func NewMemberListCommand() *cobra.Command {
	cc := &cobra.Command{
		Use:   "list",
		Short: "list is used to list all members in the cluster",

		Run: memberListCommandFunc,
	}

	return cc
}

// memberAddCommandFunc executes the "member add" command.
func memberAddCommandFunc(cmd *cobra.Command, args []string) {
	if len(args) != 1 {
		ExitWithError(ExitBadArgs, fmt.Errorf("member name not provided."))
	}

	if len(memberPeerURLs) == 0 {
		ExitWithError(ExitBadArgs, fmt.Errorf("member peer urls not provided."))
	}

	urls := strings.Split(memberPeerURLs, ",")

	resp, err := mustClientFromCmd(cmd).MemberAdd(context.TODO(), urls)
	if err != nil {
		ExitWithError(ExitError, err)
	}

	fmt.Printf("Member %16x added to cluster %16x\n", resp.Member.ID, resp.Header.ClusterId)
}

// memberRemoveCommandFunc executes the "member remove" command.
func memberRemoveCommandFunc(cmd *cobra.Command, args []string) {
	if len(args) != 1 {
		ExitWithError(ExitBadArgs, fmt.Errorf("member ID is not provided"))
	}

	id, err := strconv.ParseUint(args[0], 16, 64)
	if err != nil {
		ExitWithError(ExitBadArgs, fmt.Errorf("bad member ID arg (%v), expecting ID in Hex", err))
	}

	resp, err := mustClientFromCmd(cmd).MemberRemove(context.TODO(), id)
	if err != nil {
		ExitWithError(ExitError, err)
	}

	fmt.Printf("Member %16x removed from cluster %16x\n", id, resp.Header.ClusterId)
}

// memberUpdateCommandFunc executes the "member update" command.
func memberUpdateCommandFunc(cmd *cobra.Command, args []string) {
	if len(args) != 1 {
		ExitWithError(ExitBadArgs, fmt.Errorf("member ID is not provided"))
	}

	id, err := strconv.ParseUint(args[0], 16, 64)
	if err != nil {
		ExitWithError(ExitBadArgs, fmt.Errorf("bad member ID arg (%v), expecting ID in Hex", err))
	}

	if len(memberPeerURLs) == 0 {
		ExitWithError(ExitBadArgs, fmt.Errorf("member peer urls not provided."))
	}

	urls := strings.Split(memberPeerURLs, ",")

	resp, err := mustClientFromCmd(cmd).MemberUpdate(context.TODO(), id, urls)
	if err != nil {
		ExitWithError(ExitError, err)
	}

	fmt.Printf("Member %16x updated in cluster %16x\n", id, resp.Header.ClusterId)
}

// memberListCommandFunc executes the "member list" command.
func memberListCommandFunc(cmd *cobra.Command, args []string) {
	resp, err := mustClientFromCmd(cmd).MemberList(context.TODO())
	if err != nil {
		ExitWithError(ExitError, err)
	}

	display.MemberList(*resp)
}
