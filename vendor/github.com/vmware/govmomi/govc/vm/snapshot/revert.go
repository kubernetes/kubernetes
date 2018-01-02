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

package snapshot

import (
	"context"
	"flag"

	"github.com/vmware/govmomi/govc/cli"
	"github.com/vmware/govmomi/govc/flags"
	"github.com/vmware/govmomi/object"
)

type revert struct {
	*flags.VirtualMachineFlag

	suppressPowerOn bool
}

func init() {
	cli.Register("snapshot.revert", &revert{})
}

func (cmd *revert) Register(ctx context.Context, f *flag.FlagSet) {
	cmd.VirtualMachineFlag, ctx = flags.NewVirtualMachineFlag(ctx)
	cmd.VirtualMachineFlag.Register(ctx, f)

	f.BoolVar(&cmd.suppressPowerOn, "s", false, "Suppress power on")
}

func (cmd *revert) Usage() string {
	return "[NAME]"
}

func (cmd *revert) Description() string {
	return `Revert to snapshot of VM with given NAME.

If NAME is not provided, revert to the current snapshot.
Otherwise, NAME can be the snapshot name, tree path or moid.

Examples:
  govc snapshot.revert -vm my-vm happy-vm-state`
}

func (cmd *revert) Process(ctx context.Context) error {
	if err := cmd.VirtualMachineFlag.Process(ctx); err != nil {
		return err
	}
	return nil
}

func (cmd *revert) Run(ctx context.Context, f *flag.FlagSet) error {
	if f.NArg() > 1 {
		return flag.ErrHelp
	}

	vm, err := cmd.VirtualMachine()
	if err != nil {
		return err
	}

	if vm == nil {
		return flag.ErrHelp
	}

	var task *object.Task

	if f.NArg() == 1 {
		task, err = vm.RevertToSnapshot(ctx, f.Arg(0), cmd.suppressPowerOn)
	} else {
		task, err = vm.RevertToCurrentSnapshot(ctx, cmd.suppressPowerOn)
	}

	if err != nil {
		return err
	}

	return task.Wait(ctx)
}
