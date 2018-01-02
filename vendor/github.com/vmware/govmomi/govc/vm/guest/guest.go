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

package guest

import (
	"errors"
	"flag"

	"context"
	"net/url"

	"github.com/vmware/govmomi/govc/flags"
	"github.com/vmware/govmomi/guest"
	"github.com/vmware/govmomi/object"
)

type GuestFlag struct {
	*flags.ClientFlag
	*flags.VirtualMachineFlag

	*AuthFlag
}

func newGuestFlag(ctx context.Context) (*GuestFlag, context.Context) {
	f := &GuestFlag{}
	f.ClientFlag, ctx = flags.NewClientFlag(ctx)
	f.VirtualMachineFlag, ctx = flags.NewVirtualMachineFlag(ctx)
	f.AuthFlag, ctx = newAuthFlag(ctx)
	return f, ctx
}

func (flag *GuestFlag) Register(ctx context.Context, f *flag.FlagSet) {
	flag.ClientFlag.Register(ctx, f)
	flag.VirtualMachineFlag.Register(ctx, f)
	flag.AuthFlag.Register(ctx, f)
}

func (flag *GuestFlag) Process(ctx context.Context) error {
	if err := flag.ClientFlag.Process(ctx); err != nil {
		return err
	}
	if err := flag.VirtualMachineFlag.Process(ctx); err != nil {
		return err
	}
	if err := flag.AuthFlag.Process(ctx); err != nil {
		return err
	}
	return nil
}

func (flag *GuestFlag) FileManager() (*guest.FileManager, error) {
	ctx := context.TODO()
	c, err := flag.Client()
	if err != nil {
		return nil, err
	}

	vm, err := flag.VirtualMachine()
	if err != nil {
		return nil, err
	}

	o := guest.NewOperationsManager(c, vm.Reference())
	return o.FileManager(ctx)
}

func (flag *GuestFlag) ProcessManager() (*guest.ProcessManager, error) {
	ctx := context.TODO()
	c, err := flag.Client()
	if err != nil {
		return nil, err
	}

	vm, err := flag.VirtualMachine()
	if err != nil {
		return nil, err
	}

	o := guest.NewOperationsManager(c, vm.Reference())
	return o.ProcessManager(ctx)
}

func (flag *GuestFlag) ParseURL(urlStr string) (*url.URL, error) {
	c, err := flag.Client()
	if err != nil {
		return nil, err
	}

	return c.Client.ParseURL(urlStr)
}

func (flag *GuestFlag) VirtualMachine() (*object.VirtualMachine, error) {
	vm, err := flag.VirtualMachineFlag.VirtualMachine()
	if err != nil {
		return nil, err
	}
	if vm == nil {
		return nil, errors.New("no vm specified")
	}
	return vm, nil
}
