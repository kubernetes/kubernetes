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
	"os"
	"strconv"

	v3 "github.com/coreos/etcd/clientv3"
	"github.com/spf13/cobra"
	"golang.org/x/net/context"
)

// NewLeaseCommand returns the cobra command for "lease".
func NewLeaseCommand() *cobra.Command {
	lc := &cobra.Command{
		Use:   "lease",
		Short: "lease is used to manage leases.",
	}

	lc.AddCommand(NewLeaseCreateCommand())
	lc.AddCommand(NewLeaseRevokeCommand())
	lc.AddCommand(NewLeaseKeepAliveCommand())

	return lc
}

// NewLeaseCreateCommand returns the cobra command for "lease create".
func NewLeaseCreateCommand() *cobra.Command {
	lc := &cobra.Command{
		Use:   "create",
		Short: "create is used to create leases.",

		Run: leaseCreateCommandFunc,
	}

	return lc
}

// leaseCreateCommandFunc executes the "lease create" command.
func leaseCreateCommandFunc(cmd *cobra.Command, args []string) {
	if len(args) != 1 {
		ExitWithError(ExitBadArgs, fmt.Errorf("lease create command needs TTL argument."))
	}

	ttl, err := strconv.ParseInt(args[0], 10, 64)
	if err != nil {
		ExitWithError(ExitBadArgs, fmt.Errorf("bad TTL (%v)", err))
	}

	resp, err := mustClientFromCmd(cmd).Create(context.TODO(), ttl)
	if err != nil {
		fmt.Fprintf(os.Stderr, "failed to create lease (%v)\n", err)
		return
	}
	fmt.Printf("lease %016x created with TTL(%ds)\n", resp.ID, resp.TTL)
}

// NewLeaseRevokeCommand returns the cobra command for "lease revoke".
func NewLeaseRevokeCommand() *cobra.Command {
	lc := &cobra.Command{
		Use:   "revoke",
		Short: "revoke is used to revoke leases.",

		Run: leaseRevokeCommandFunc,
	}

	return lc
}

// leaseRevokeCommandFunc executes the "lease create" command.
func leaseRevokeCommandFunc(cmd *cobra.Command, args []string) {
	if len(args) != 1 {
		ExitWithError(ExitBadArgs, fmt.Errorf("lease revoke command needs 1 argument"))
	}

	id, err := strconv.ParseInt(args[0], 16, 64)
	if err != nil {
		ExitWithError(ExitBadArgs, fmt.Errorf("bad lease ID arg (%v), expecting ID in Hex", err))
	}

	_, err = mustClientFromCmd(cmd).Revoke(context.TODO(), v3.LeaseID(id))
	if err != nil {
		fmt.Fprintf(os.Stderr, "failed to revoke lease (%v)\n", err)
		return
	}
	fmt.Printf("lease %016x revoked\n", id)
}

// NewLeaseKeepAliveCommand returns the cobra command for "lease keep-alive".
func NewLeaseKeepAliveCommand() *cobra.Command {
	lc := &cobra.Command{
		Use:   "keep-alive",
		Short: "keep-alive is used to keep leases alive.",

		Run: leaseKeepAliveCommandFunc,
	}

	return lc
}

// leaseKeepAliveCommandFunc executes the "lease keep-alive" command.
func leaseKeepAliveCommandFunc(cmd *cobra.Command, args []string) {
	if len(args) != 1 {
		ExitWithError(ExitBadArgs, fmt.Errorf("lease keep-alive command needs lease ID as argument"))
	}

	id, err := strconv.ParseInt(args[0], 16, 64)
	if err != nil {
		ExitWithError(ExitBadArgs, fmt.Errorf("bad lease ID arg (%v), expecting ID in Hex", err))
	}

	respc, kerr := mustClientFromCmd(cmd).KeepAlive(context.TODO(), v3.LeaseID(id))
	if kerr != nil {
		ExitWithError(ExitBadConnection, kerr)
	}

	for resp := range respc {
		fmt.Printf("lease %016x keepalived with TTL(%d)\n", resp.ID, resp.TTL)
	}
	fmt.Printf("lease %016x expired or revoked.\n", id)
}
