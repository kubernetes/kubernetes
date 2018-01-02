/*
Copyright (c) 2014-2016 VMware, Inc. All Rights Reserved.

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
	"github.com/vmware/govmomi/vim25/soap"
	"github.com/vmware/govmomi/vim25/types"
)

type power struct {
	*flags.ClientFlag
	*flags.SearchFlag

	On       bool
	Off      bool
	Reset    bool
	Reboot   bool
	Shutdown bool
	Suspend  bool
	Force    bool
}

func init() {
	cli.Register("vm.power", &power{})
}

func (cmd *power) Register(ctx context.Context, f *flag.FlagSet) {
	cmd.ClientFlag, ctx = flags.NewClientFlag(ctx)
	cmd.ClientFlag.Register(ctx, f)

	cmd.SearchFlag, ctx = flags.NewSearchFlag(ctx, flags.SearchVirtualMachines)
	cmd.SearchFlag.Register(ctx, f)

	f.BoolVar(&cmd.On, "on", false, "Power on")
	f.BoolVar(&cmd.Off, "off", false, "Power off")
	f.BoolVar(&cmd.Reset, "reset", false, "Power reset")
	f.BoolVar(&cmd.Suspend, "suspend", false, "Power suspend")
	f.BoolVar(&cmd.Reboot, "r", false, "Reboot guest")
	f.BoolVar(&cmd.Shutdown, "s", false, "Shutdown guest")
	f.BoolVar(&cmd.Force, "force", false, "Force (ignore state error and hard shutdown/reboot if tools unavailable)")
}

func (cmd *power) Process(ctx context.Context) error {
	if err := cmd.ClientFlag.Process(ctx); err != nil {
		return err
	}
	if err := cmd.SearchFlag.Process(ctx); err != nil {
		return err
	}
	opts := []bool{cmd.On, cmd.Off, cmd.Reset, cmd.Suspend, cmd.Reboot, cmd.Shutdown}
	selected := false

	for _, opt := range opts {
		if opt {
			if selected {
				return flag.ErrHelp
			}
			selected = opt
		}
	}

	if !selected {
		return flag.ErrHelp
	}

	return nil
}

func isToolsUnavailable(err error) bool {
	if soap.IsSoapFault(err) {
		soapFault := soap.ToSoapFault(err)
		if _, ok := soapFault.VimFault().(types.ToolsUnavailable); ok {
			return ok
		}
	}

	return false
}

func (cmd *power) Run(ctx context.Context, f *flag.FlagSet) error {
	vms, err := cmd.VirtualMachines(f.Args())
	if err != nil {
		return err
	}

	for _, vm := range vms {
		var task *object.Task

		switch {
		case cmd.On:
			fmt.Fprintf(cmd, "Powering on %s... ", vm.Reference())
			task, err = vm.PowerOn(ctx)
		case cmd.Off:
			fmt.Fprintf(cmd, "Powering off %s... ", vm.Reference())
			task, err = vm.PowerOff(ctx)
		case cmd.Reset:
			fmt.Fprintf(cmd, "Reset %s... ", vm.Reference())
			task, err = vm.Reset(ctx)
		case cmd.Suspend:
			fmt.Fprintf(cmd, "Suspend %s... ", vm.Reference())
			task, err = vm.Suspend(ctx)
		case cmd.Reboot:
			fmt.Fprintf(cmd, "Reboot guest %s... ", vm.Reference())
			err = vm.RebootGuest(ctx)

			if err != nil && cmd.Force && isToolsUnavailable(err) {
				task, err = vm.Reset(ctx)
			}
		case cmd.Shutdown:
			fmt.Fprintf(cmd, "Shutdown guest %s... ", vm.Reference())
			err = vm.ShutdownGuest(ctx)

			if err != nil && cmd.Force && isToolsUnavailable(err) {
				task, err = vm.PowerOff(ctx)
			}
		}

		if err != nil {
			return err
		}

		if task != nil {
			err = task.Wait(ctx)
		}
		if err == nil {
			fmt.Fprintf(cmd, "OK\n")
			continue
		}

		if cmd.Force {
			fmt.Fprintf(cmd, "Error: %s\n", err)
			continue
		}

		return err
	}

	return nil
}
