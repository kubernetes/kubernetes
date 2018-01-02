/*
Copyright (c) 2015 VMware, Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package host

import (
	"context"
	"flag"
	"fmt"

	"github.com/vmware/govmomi/govc/cli"
	"github.com/vmware/govmomi/govc/flags"
	"github.com/vmware/govmomi/object"
	"github.com/vmware/govmomi/vim25/types"
)

type reconnect struct {
	*flags.HostSystemFlag
	*flags.HostConnectFlag

	types.HostSystemReconnectSpec
}

func init() {
	cli.Register("host.reconnect", &reconnect{})
}

func (cmd *reconnect) Register(ctx context.Context, f *flag.FlagSet) {
	cmd.HostSystemFlag, ctx = flags.NewHostSystemFlag(ctx)
	cmd.HostSystemFlag.Register(ctx, f)

	cmd.HostConnectFlag, ctx = flags.NewHostConnectFlag(ctx)
	cmd.HostConnectFlag.Register(ctx, f)

	cmd.HostSystemReconnectSpec.SyncState = types.NewBool(false)
	f.BoolVar(cmd.HostSystemReconnectSpec.SyncState, "sync-state", false, "Sync state")
}

func (cmd *reconnect) Process(ctx context.Context) error {
	if err := cmd.HostSystemFlag.Process(ctx); err != nil {
		return err
	}
	if err := cmd.HostConnectFlag.Process(ctx); err != nil {
		return err
	}
	return nil
}

func (cmd *reconnect) Description() string {
	return `Reconnect HOST to vCenter.

This command can also be used to change connection properties (hostname, fingerprint, username, password),
without disconnecting the host.`
}

func (cmd *reconnect) Reconnect(ctx context.Context, host *object.HostSystem) error {
	task, err := host.Reconnect(ctx, &cmd.HostConnectSpec, &cmd.HostSystemReconnectSpec)
	if err != nil {
		return err
	}

	logger := cmd.ProgressLogger(fmt.Sprintf("%s reconnecting... ", host.InventoryPath))
	defer logger.Wait()

	_, err = task.WaitForResult(ctx, logger)
	return err
}

func (cmd *reconnect) Run(ctx context.Context, f *flag.FlagSet) error {
	hosts, err := cmd.HostSystems(f.Args())
	if err != nil {
		return err
	}

	for _, host := range hosts {
		err = cmd.Reconnect(ctx, host)
		if err != nil {
			return err
		}
	}

	return nil
}
