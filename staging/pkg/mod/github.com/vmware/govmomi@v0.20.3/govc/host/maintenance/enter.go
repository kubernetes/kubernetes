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

type enter struct {
	*flags.HostSystemFlag

	timeout  int32
	evacuate bool
}

func init() {
	cli.Register("host.maintenance.enter", &enter{})
}

func (cmd *enter) Register(ctx context.Context, f *flag.FlagSet) {
	cmd.HostSystemFlag, ctx = flags.NewHostSystemFlag(ctx)
	cmd.HostSystemFlag.Register(ctx, f)

	f.Var(flags.NewInt32(&cmd.timeout), "timeout", "Timeout")
	f.BoolVar(&cmd.evacuate, "evacuate", false, "Evacuate powered off VMs")
}

func (cmd *enter) Process(ctx context.Context) error {
	if err := cmd.HostSystemFlag.Process(ctx); err != nil {
		return err
	}
	return nil
}

func (cmd *enter) Usage() string {
	return "HOST..."
}

func (cmd *enter) Description() string {
	return `Put HOST in maintenance mode.

While this task is running and when the host is in maintenance mode,
no VMs can be powered on and no provisioning operations can be performed on the host.`
}

func (cmd *enter) EnterMaintenanceMode(ctx context.Context, host *object.HostSystem) error {
	task, err := host.EnterMaintenanceMode(ctx, cmd.timeout, cmd.evacuate, nil) // TODO: spec param
	if err != nil {
		return err
	}

	logger := cmd.ProgressLogger(fmt.Sprintf("%s entering maintenance mode... ", host.InventoryPath))
	defer logger.Wait()

	_, err = task.WaitForResult(ctx, logger)
	return err
}

func (cmd *enter) Run(ctx context.Context, f *flag.FlagSet) error {
	hosts, err := cmd.HostSystems(f.Args())
	if err != nil {
		return err
	}

	for _, host := range hosts {
		err = cmd.EnterMaintenanceMode(ctx, host)
		if err != nil {
			return err
		}
	}

	return nil
}
