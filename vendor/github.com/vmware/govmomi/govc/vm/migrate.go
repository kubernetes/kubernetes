/*
Copyright (c) 2016 VMware, Inc. All Rights Reserved.

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

package vm

import (
	"context"
	"flag"
	"fmt"

	"github.com/vmware/govmomi/govc/cli"
	"github.com/vmware/govmomi/govc/flags"
	"github.com/vmware/govmomi/object"
	"github.com/vmware/govmomi/vim25/types"
)

type migrate struct {
	*flags.ResourcePoolFlag
	*flags.HostSystemFlag
	*flags.DatastoreFlag
	*flags.SearchFlag

	priority types.VirtualMachineMovePriority
	spec     types.VirtualMachineRelocateSpec
}

func init() {
	cli.Register("vm.migrate", &migrate{})
}

func (cmd *migrate) Register(ctx context.Context, f *flag.FlagSet) {
	cmd.SearchFlag, ctx = flags.NewSearchFlag(ctx, flags.SearchVirtualMachines)
	cmd.SearchFlag.Register(ctx, f)

	cmd.ResourcePoolFlag, ctx = flags.NewResourcePoolFlag(ctx)
	cmd.ResourcePoolFlag.Register(ctx, f)

	cmd.HostSystemFlag, ctx = flags.NewHostSystemFlag(ctx)
	cmd.HostSystemFlag.Register(ctx, f)

	cmd.DatastoreFlag, ctx = flags.NewDatastoreFlag(ctx)
	cmd.DatastoreFlag.Register(ctx, f)

	f.StringVar((*string)(&cmd.priority), "priority", string(types.VirtualMachineMovePriorityDefaultPriority), "The task priority")
}

func (cmd *migrate) Process(ctx context.Context) error {
	if err := cmd.ResourcePoolFlag.Process(ctx); err != nil {
		return err
	}
	if err := cmd.HostSystemFlag.Process(ctx); err != nil {
		return err
	}
	if err := cmd.DatastoreFlag.Process(ctx); err != nil {
		return err
	}

	return nil
}

func (cmd *migrate) Usage() string {
	return "VM..."
}

func (cmd *migrate) Description() string {
	return `Migrates VM to a specific resource pool, host or datastore.

Examples:
  govc vm.migrate -host another-host vm-1 vm-2 vm-3
  govc vm.migrate -ds another-ds vm-1 vm-2 vm-3`
}

func (cmd *migrate) relocate(ctx context.Context, vm *object.VirtualMachine) error {
	task, err := vm.Relocate(ctx, cmd.spec, cmd.priority)
	if err != nil {
		return err
	}

	logger := cmd.DatastoreFlag.ProgressLogger(fmt.Sprintf("migrating %s... ", vm.Reference()))
	_, err = task.WaitForResult(ctx, logger)
	if err != nil {
		return err
	}

	logger.Wait()

	return nil
}

func (cmd *migrate) Run(ctx context.Context, f *flag.FlagSet) error {
	vms, err := cmd.VirtualMachines(f.Args())
	if err != nil {
		return err
	}

	host, err := cmd.HostSystemFlag.HostSystemIfSpecified()
	if err != nil {
		return err
	}

	if host != nil {
		ref := host.Reference()
		cmd.spec.Host = &ref
	}

	pool, err := cmd.ResourcePoolFlag.ResourcePoolIfSpecified()
	if err != nil {
		return err
	}

	if pool != nil {
		ref := pool.Reference()
		cmd.spec.Pool = &ref
	}

	ds, err := cmd.DatastoreFlag.DatastoreIfSpecified()
	if err != nil {
		return err
	}

	if ds != nil {
		ref := ds.Reference()
		cmd.spec.Datastore = &ref
	}

	for _, vm := range vms {
		err = cmd.relocate(ctx, vm)
		if err != nil {
			return err
		}
	}

	return nil
}
