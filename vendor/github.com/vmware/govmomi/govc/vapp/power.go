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

package vapp

import (
	"context"
	"flag"
	"fmt"

	"github.com/vmware/govmomi/govc/cli"
	"github.com/vmware/govmomi/govc/flags"
	"github.com/vmware/govmomi/object"
)

type power struct {
	*flags.SearchFlag

	On      bool
	Off     bool
	Suspend bool
	Force   bool
}

func init() {
	cli.Register("vapp.power", &power{})
}

func (cmd *power) Register(ctx context.Context, f *flag.FlagSet) {
	cmd.SearchFlag, ctx = flags.NewSearchFlag(ctx, flags.SearchVirtualApps)
	cmd.SearchFlag.Register(ctx, f)

	f.BoolVar(&cmd.On, "on", false, "Power on")
	f.BoolVar(&cmd.Off, "off", false, "Power off")
	f.BoolVar(&cmd.Suspend, "suspend", false, "Power suspend")
	f.BoolVar(&cmd.Force, "force", false, "Force (If force is false, the shutdown order in the vApp is executed. If force is true, all virtual machines are powered-off (regardless of shutdown order))")
}

func (cmd *power) Process(ctx context.Context) error {
	if err := cmd.SearchFlag.Process(ctx); err != nil {
		return err
	}
	opts := []bool{cmd.On, cmd.Off, cmd.Suspend}
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

func (cmd *power) Run(ctx context.Context, f *flag.FlagSet) error {
	vapps, err := cmd.VirtualApps(f.Args())
	if err != nil {
		return err
	}

	for _, vapp := range vapps {
		var task *object.Task

		switch {
		case cmd.On:
			fmt.Fprintf(cmd, "Powering on %s... ", vapp.Reference())
			task, err = vapp.PowerOn(ctx)
		case cmd.Off:
			fmt.Fprintf(cmd, "Powering off %s... ", vapp.Reference())
			task, err = vapp.PowerOff(ctx, cmd.Force)
		case cmd.Suspend:
			fmt.Fprintf(cmd, "Suspend %s... ", vapp.Reference())
			task, err = vapp.Suspend(ctx)
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

		return err
	}

	return nil
}
