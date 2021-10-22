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

package maintenance

import (
	"context"
	"flag"
	"fmt"

	"github.com/vmware/govmomi/govc/cli"
	"github.com/vmware/govmomi/govc/flags"
	"github.com/vmware/govmomi/object"
)

type exit struct {
	*flags.HostSystemFlag

	timeout int32
}

func init() {
	cli.Register("host.maintenance.exit", &exit{})
}

func (cmd *exit) Register(ctx context.Context, f *flag.FlagSet) {
	cmd.HostSystemFlag, ctx = flags.NewHostSystemFlag(ctx)
	cmd.HostSystemFlag.Register(ctx, f)

	f.Var(flags.NewInt32(&cmd.timeout), "timeout", "Timeout")
}

func (cmd *exit) Process(ctx context.Context) error {
	if err := cmd.HostSystemFlag.Process(ctx); err != nil {
		return err
	}
	return nil
}

func (cmd *exit) Usage() string {
	return "HOST..."
}

func (cmd *exit) Description() string {
	return `Take HOST out of maintenance mode.

This blocks if any concurrent running maintenance-only host configurations operations are being performed.
For example, if VMFS volumes are being upgraded.

The 'timeout' flag is the number of seconds to wait for the exit maintenance mode to succeed.
If the timeout is less than or equal to zero, there is no timeout.`
}

func (cmd *exit) ExitMaintenanceMode(ctx context.Context, host *object.HostSystem) error {
	task, err := host.ExitMaintenanceMode(ctx, cmd.timeout)
	if err != nil {
		return err
	}

	logger := cmd.ProgressLogger(fmt.Sprintf("%s exiting maintenance mode... ", host.InventoryPath))
	defer logger.Wait()

	_, err = task.WaitForResult(ctx, logger)
	return err
}

func (cmd *exit) Run(ctx context.Context, f *flag.FlagSet) error {
	hosts, err := cmd.HostSystems(f.Args())
	if err != nil {
		return err
	}

	for _, host := range hosts {
		err = cmd.ExitMaintenanceMode(ctx, host)
		if err != nil {
			return err
		}
	}

	return nil
}
