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
	"strconv"

	v3 "github.com/coreos/etcd/clientv3"
	"github.com/spf13/cobra"
	"golang.org/x/net/context"
)

// NewLeaseCommand returns the cobra command for "lease".
func NewLeaseCommand() *cobra.Command {
	lc := &cobra.Command{
		Use:   "lease <subcommand>",
		Short: "Lease related commands",
	}

	lc.AddCommand(NewLeaseGrantCommand())
	lc.AddCommand(NewLeaseRevokeCommand())
	lc.AddCommand(NewLeaseTimeToLiveCommand())
	lc.AddCommand(NewLeaseKeepAliveCommand())

	return lc
}

// NewLeaseGrantCommand returns the cobra command for "lease grant".
func NewLeaseGrantCommand() *cobra.Command {
	lc := &cobra.Command{
		Use:   "grant <ttl>",
		Short: "Creates leases",

		Run: leaseGrantCommandFunc,
	}

	return lc
}

// leaseGrantCommandFunc executes the "lease grant" command.
func leaseGrantCommandFunc(cmd *cobra.Command, args []string) {
	if len(args) != 1 {
		ExitWithError(ExitBadArgs, fmt.Errorf("lease grant command needs TTL argument."))
	}

	ttl, err := strconv.ParseInt(args[0], 10, 64)
	if err != nil {
		ExitWithError(ExitBadArgs, fmt.Errorf("bad TTL (%v)", err))
	}

	ctx, cancel := commandCtx(cmd)
	resp, err := mustClientFromCmd(cmd).Grant(ctx, ttl)
	cancel()
	if err != nil {
		ExitWithError(ExitError, fmt.Errorf("failed to grant lease (%v)\n", err))
	}
	display.Grant(*resp)
}

// NewLeaseRevokeCommand returns the cobra command for "lease revoke".
func NewLeaseRevokeCommand() *cobra.Command {
	lc := &cobra.Command{
		Use:   "revoke <leaseID>",
		Short: "Revokes leases",

		Run: leaseRevokeCommandFunc,
	}

	return lc
}

// leaseRevokeCommandFunc executes the "lease grant" command.
func leaseRevokeCommandFunc(cmd *cobra.Command, args []string) {
	if len(args) != 1 {
		ExitWithError(ExitBadArgs, fmt.Errorf("lease revoke command needs 1 argument"))
	}

	id := leaseFromArgs(args[0])
	ctx, cancel := commandCtx(cmd)
	resp, err := mustClientFromCmd(cmd).Revoke(ctx, id)
	cancel()
	if err != nil {
		ExitWithError(ExitError, fmt.Errorf("failed to revoke lease (%v)\n", err))
	}
	display.Revoke(id, *resp)
}

var timeToLiveKeys bool

// NewLeaseTimeToLiveCommand returns the cobra command for "lease timetolive".
func NewLeaseTimeToLiveCommand() *cobra.Command {
	lc := &cobra.Command{
		Use:   "timetolive <leaseID> [options]",
		Short: "Get lease information",

		Run: leaseTimeToLiveCommandFunc,
	}
	lc.Flags().BoolVar(&timeToLiveKeys, "keys", false, "Get keys attached to this lease")

	return lc
}

// leaseTimeToLiveCommandFunc executes the "lease timetolive" command.
func leaseTimeToLiveCommandFunc(cmd *cobra.Command, args []string) {
	if len(args) != 1 {
		ExitWithError(ExitBadArgs, fmt.Errorf("lease timetolive command needs lease ID as argument"))
	}
	var opts []v3.LeaseOption
	if timeToLiveKeys {
		opts = append(opts, v3.WithAttachedKeys())
	}
	resp, rerr := mustClientFromCmd(cmd).TimeToLive(context.TODO(), leaseFromArgs(args[0]), opts...)
	if rerr != nil {
		ExitWithError(ExitBadConnection, rerr)
	}
	display.TimeToLive(*resp, timeToLiveKeys)
}

// NewLeaseKeepAliveCommand returns the cobra command for "lease keep-alive".
func NewLeaseKeepAliveCommand() *cobra.Command {
	lc := &cobra.Command{
		Use:   "keep-alive <leaseID>",
		Short: "Keeps leases alive (renew)",

		Run: leaseKeepAliveCommandFunc,
	}

	return lc
}

// leaseKeepAliveCommandFunc executes the "lease keep-alive" command.
func leaseKeepAliveCommandFunc(cmd *cobra.Command, args []string) {
	if len(args) != 1 {
		ExitWithError(ExitBadArgs, fmt.Errorf("lease keep-alive command needs lease ID as argument"))
	}

	id := leaseFromArgs(args[0])
	respc, kerr := mustClientFromCmd(cmd).KeepAlive(context.TODO(), id)
	if kerr != nil {
		ExitWithError(ExitBadConnection, kerr)
	}

	for resp := range respc {
		display.KeepAlive(*resp)
	}

	if _, ok := (display).(*simplePrinter); ok {
		fmt.Printf("lease %016x expired or revoked.\n", id)
	}
}

func leaseFromArgs(arg string) v3.LeaseID {
	id, err := strconv.ParseInt(arg, 16, 64)
	if err != nil {
		ExitWithError(ExitBadArgs, fmt.Errorf("bad lease ID arg (%v), expecting ID in Hex", err))
	}
	return v3.LeaseID(id)
}
