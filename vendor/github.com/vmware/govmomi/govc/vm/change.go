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

package vm

import (
	"context"
	"flag"
	"fmt"
	"strings"

	"github.com/vmware/govmomi/govc/cli"
	"github.com/vmware/govmomi/govc/flags"
	"github.com/vmware/govmomi/vim25/types"
)

type extraConfig []types.BaseOptionValue

func (e *extraConfig) String() string {
	return fmt.Sprintf("%v", *e)
}

func (e *extraConfig) Set(v string) error {
	r := strings.SplitN(v, "=", 2)
	if len(r) < 2 {
		return fmt.Errorf("failed to parse extraConfig: %s", v)
	} else if r[1] == "" {
		return fmt.Errorf("empty value: %s", v)
	}
	*e = append(*e, &types.OptionValue{Key: r[0], Value: r[1]})
	return nil
}

type change struct {
	*flags.VirtualMachineFlag

	types.VirtualMachineConfigSpec
	extraConfig extraConfig
}

func init() {
	cli.Register("vm.change", &change{})
}

func (cmd *change) Register(ctx context.Context, f *flag.FlagSet) {
	cmd.VirtualMachineFlag, ctx = flags.NewVirtualMachineFlag(ctx)
	cmd.VirtualMachineFlag.Register(ctx, f)

	f.Int64Var(&cmd.MemoryMB, "m", 0, "Size in MB of memory")
	f.Var(flags.NewInt32(&cmd.NumCPUs), "c", "Number of CPUs")
	f.StringVar(&cmd.GuestId, "g", "", "Guest OS")
	f.StringVar(&cmd.Name, "name", "", "Display name")
	f.Var(&cmd.extraConfig, "e", "ExtraConfig. <key>=<value>")

	f.Var(flags.NewOptionalBool(&cmd.NestedHVEnabled), "nested-hv-enabled", "Enable nested hardware-assisted virtualization")
	cmd.Tools = &types.ToolsConfigInfo{}
	f.Var(flags.NewOptionalBool(&cmd.Tools.SyncTimeWithHost), "sync-time-with-host", "Enable SyncTimeWithHost")
}

func (cmd *change) Description() string {
	return `Change VM configuration.

Examples:
  govc vm.change -vm $vm -e smc.present=TRUE -e ich7m.present=TRUE`
}

func (cmd *change) Process(ctx context.Context) error {
	if err := cmd.VirtualMachineFlag.Process(ctx); err != nil {
		return err
	}
	return nil
}

func (cmd *change) Run(ctx context.Context, f *flag.FlagSet) error {
	vm, err := cmd.VirtualMachine()
	if err != nil {
		return err
	}

	if vm == nil {
		return flag.ErrHelp
	}

	if len(cmd.extraConfig) > 0 {
		cmd.VirtualMachineConfigSpec.ExtraConfig = cmd.extraConfig
	}

	task, err := vm.Reconfigure(ctx, cmd.VirtualMachineConfigSpec)
	if err != nil {
		return err
	}

	return task.Wait(ctx)
}
