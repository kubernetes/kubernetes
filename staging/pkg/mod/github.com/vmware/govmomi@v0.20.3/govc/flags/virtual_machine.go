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

package flags

import (
	"context"
	"flag"
	"fmt"
	"os"

	"github.com/vmware/govmomi/object"
)

type VirtualMachineFlag struct {
	common

	*ClientFlag
	*DatacenterFlag
	*SearchFlag

	name string
	vm   *object.VirtualMachine
}

var virtualMachineFlagKey = flagKey("virtualMachine")

func NewVirtualMachineFlag(ctx context.Context) (*VirtualMachineFlag, context.Context) {
	if v := ctx.Value(virtualMachineFlagKey); v != nil {
		return v.(*VirtualMachineFlag), ctx
	}

	v := &VirtualMachineFlag{}
	v.ClientFlag, ctx = NewClientFlag(ctx)
	v.DatacenterFlag, ctx = NewDatacenterFlag(ctx)
	v.SearchFlag, ctx = NewSearchFlag(ctx, SearchVirtualMachines)
	ctx = context.WithValue(ctx, virtualMachineFlagKey, v)
	return v, ctx
}

func (flag *VirtualMachineFlag) Register(ctx context.Context, f *flag.FlagSet) {
	flag.RegisterOnce(func() {
		flag.ClientFlag.Register(ctx, f)
		flag.DatacenterFlag.Register(ctx, f)
		flag.SearchFlag.Register(ctx, f)

		env := "GOVC_VM"
		value := os.Getenv(env)
		usage := fmt.Sprintf("Virtual machine [%s]", env)
		f.StringVar(&flag.name, "vm", value, usage)
	})
}

func (flag *VirtualMachineFlag) Process(ctx context.Context) error {
	return flag.ProcessOnce(func() error {
		if err := flag.ClientFlag.Process(ctx); err != nil {
			return err
		}
		if err := flag.DatacenterFlag.Process(ctx); err != nil {
			return err
		}
		if err := flag.SearchFlag.Process(ctx); err != nil {
			return err
		}
		return nil
	})
}

func (flag *VirtualMachineFlag) VirtualMachine() (*object.VirtualMachine, error) {
	ctx := context.TODO()

	if flag.vm != nil {
		return flag.vm, nil
	}

	// Use search flags if specified.
	if flag.SearchFlag.IsSet() {
		vm, err := flag.SearchFlag.VirtualMachine()
		if err != nil {
			return nil, err
		}

		flag.vm = vm
		return flag.vm, nil
	}

	// Never look for a default virtual machine.
	if flag.name == "" {
		return nil, nil
	}

	finder, err := flag.Finder()
	if err != nil {
		return nil, err
	}

	flag.vm, err = finder.VirtualMachine(ctx, flag.name)
	return flag.vm, err
}
