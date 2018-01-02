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

	"github.com/vmware/govmomi/govc/cli"
	"github.com/vmware/govmomi/govc/flags"
	"github.com/vmware/govmomi/object"
	"github.com/vmware/govmomi/vim25"
)

type register struct {
	*flags.ClientFlag
	*flags.DatacenterFlag
	*flags.DatastoreFlag
	*flags.ResourcePoolFlag
	*flags.HostSystemFlag
	*flags.FolderFlag

	name     string
	template bool

	Client       *vim25.Client
	Datacenter   *object.Datacenter
	Datastore    *object.Datastore
	ResourcePool *object.ResourcePool
	HostSystem   *object.HostSystem
	Folder       *object.Folder
}

func init() {
	cli.Register("vm.register", &register{})
}

func (cmd *register) Register(ctx context.Context, f *flag.FlagSet) {
	cmd.ClientFlag, ctx = flags.NewClientFlag(ctx)
	cmd.ClientFlag.Register(ctx, f)

	cmd.DatacenterFlag, ctx = flags.NewDatacenterFlag(ctx)
	cmd.DatacenterFlag.Register(ctx, f)

	cmd.DatastoreFlag, ctx = flags.NewDatastoreFlag(ctx)
	cmd.DatastoreFlag.Register(ctx, f)

	cmd.ResourcePoolFlag, ctx = flags.NewResourcePoolFlag(ctx)
	cmd.ResourcePoolFlag.Register(ctx, f)

	cmd.HostSystemFlag, ctx = flags.NewHostSystemFlag(ctx)
	cmd.HostSystemFlag.Register(ctx, f)

	cmd.FolderFlag, ctx = flags.NewFolderFlag(ctx)
	cmd.FolderFlag.Register(ctx, f)

	f.StringVar(&cmd.name, "name", "", "Name of the VM")
	f.BoolVar(&cmd.template, "as-template", false, "Mark VM as template")
}

func (cmd *register) Process(ctx context.Context) error {
	if err := cmd.ClientFlag.Process(ctx); err != nil {
		return err
	}
	if err := cmd.DatacenterFlag.Process(ctx); err != nil {
		return err
	}
	if err := cmd.DatastoreFlag.Process(ctx); err != nil {
		return err
	}
	if err := cmd.ResourcePoolFlag.Process(ctx); err != nil {
		return err
	}
	if err := cmd.HostSystemFlag.Process(ctx); err != nil {
		return err
	}
	if err := cmd.FolderFlag.Process(ctx); err != nil {
		return err
	}

	return nil
}

func (cmd *register) Usage() string {
	return "VMX"
}

func (cmd *register) Description() string {
	return `Add an existing VM to the inventory.

VMX is a path to the vm config file, relative to DATASTORE.

Examples:
  govc vm.register path/name.vmx`
}

func (cmd *register) Run(ctx context.Context, f *flag.FlagSet) error {
	var err error

	if len(f.Args()) != 1 {
		return flag.ErrHelp
	}

	cmd.Client, err = cmd.ClientFlag.Client()
	if err != nil {
		return err
	}

	cmd.Datacenter, err = cmd.DatacenterFlag.Datacenter()
	if err != nil {
		return err
	}

	cmd.Datastore, err = cmd.DatastoreFlag.Datastore()
	if err != nil {
		return err
	}

	cmd.HostSystem, err = cmd.HostSystemFlag.HostSystemIfSpecified()
	if err != nil {
		return err
	}

	if cmd.HostSystem != nil {
		if cmd.ResourcePool, err = cmd.HostSystem.ResourcePool(ctx); err != nil {
			return err
		}
	} else if !cmd.template {
		if cmd.ResourcePool, err = cmd.ResourcePoolFlag.ResourcePool(); err != nil {
			return err
		}
	}

	if cmd.Folder, err = cmd.FolderFlag.Folder(); err != nil {
		return err
	}

	path := cmd.Datastore.Path(f.Arg(0))

	task, err := cmd.Folder.RegisterVM(ctx, path, cmd.name, cmd.template, cmd.ResourcePool, cmd.HostSystem)
	if err != nil {
		return err
	}

	return task.Wait(ctx)
}
