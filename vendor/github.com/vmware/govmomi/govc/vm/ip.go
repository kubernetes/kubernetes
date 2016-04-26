/*
Copyright (c) 2014-2015 VMware, Inc. All Rights Reserved.

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
	"flag"
	"fmt"
	"time"

	"github.com/vmware/govmomi/govc/cli"
	"github.com/vmware/govmomi/govc/flags"
	"github.com/vmware/govmomi/govc/host/esxcli"
	"github.com/vmware/govmomi/object"
	"golang.org/x/net/context"
)

type ip struct {
	*flags.OutputFlag
	*flags.SearchFlag

	esx bool
}

func init() {
	cli.Register("vm.ip", &ip{})
}

func (cmd *ip) Register(ctx context.Context, f *flag.FlagSet) {
	cmd.OutputFlag, ctx = flags.NewOutputFlag(ctx)
	cmd.OutputFlag.Register(ctx, f)

	cmd.SearchFlag, ctx = flags.NewSearchFlag(ctx, flags.SearchVirtualMachines)
	cmd.SearchFlag.Register(ctx, f)

	f.BoolVar(&cmd.esx, "esxcli", false, "Use esxcli instead of guest tools")
}

func (cmd *ip) Process(ctx context.Context) error {
	if err := cmd.OutputFlag.Process(ctx); err != nil {
		return err
	}
	if err := cmd.SearchFlag.Process(ctx); err != nil {
		return err
	}
	return nil
}

func (cmd *ip) Run(ctx context.Context, f *flag.FlagSet) error {
	c, err := cmd.Client()
	if err != nil {
		return err
	}

	vms, err := cmd.VirtualMachines(f.Args())
	if err != nil {
		return err
	}

	var get func(*object.VirtualMachine) (string, error)

	if cmd.esx {
		get = func(vm *object.VirtualMachine) (string, error) {
			guest := esxcli.NewGuestInfo(c)

			ticker := time.NewTicker(time.Millisecond * 500)
			defer ticker.Stop()

			for {
				select {
				case <-ticker.C:
					ip, err := guest.IpAddress(vm)
					if err != nil {
						return "", err
					}

					if ip != "0.0.0.0" {
						return ip, nil
					}
				}
			}
		}
	} else {
		get = func(vm *object.VirtualMachine) (string, error) {
			return vm.WaitForIP(context.TODO())
		}
	}

	for _, vm := range vms {
		ip, err := get(vm)
		if err != nil {
			return err
		}

		// TODO(PN): Display inventory path to VM
		fmt.Fprintf(cmd, "%s\n", ip)
	}

	return nil
}
