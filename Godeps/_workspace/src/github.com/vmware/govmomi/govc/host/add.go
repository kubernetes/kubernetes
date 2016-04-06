/*
Copyright (c) 2015-2016 VMware, Inc. All Rights Reserved.

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

package host

import (
	"flag"
	"fmt"

	"github.com/vmware/govmomi/govc/cli"
	"github.com/vmware/govmomi/govc/flags"
	"github.com/vmware/govmomi/object"
	"github.com/vmware/govmomi/vim25/methods"
	"github.com/vmware/govmomi/vim25/types"
	"golang.org/x/net/context"
)

type add struct {
	*flags.ClientFlag
	*flags.DatacenterFlag
	*flags.HostConnectFlag

	parent  string
	connect bool
}

func init() {
	cli.Register("host.add", &add{})
}

func (cmd *add) Register(ctx context.Context, f *flag.FlagSet) {
	cmd.ClientFlag, ctx = flags.NewClientFlag(ctx)
	cmd.ClientFlag.Register(ctx, f)

	cmd.DatacenterFlag, ctx = flags.NewDatacenterFlag(ctx)
	cmd.DatacenterFlag.Register(ctx, f)

	cmd.HostConnectFlag, ctx = flags.NewHostConnectFlag(ctx)
	cmd.HostConnectFlag.Register(ctx, f)

	f.StringVar(&cmd.parent, "parent", "", "Path to folder to add the host to")
	f.BoolVar(&cmd.connect, "connect", true, "Immediately connect to host")
}

func (cmd *add) Process(ctx context.Context) error {
	if err := cmd.ClientFlag.Process(ctx); err != nil {
		return err
	}
	if err := cmd.DatacenterFlag.Process(ctx); err != nil {
		return err
	}
	if err := cmd.HostConnectFlag.Process(ctx); err != nil {
		return err
	}
	if cmd.HostName == "" {
		return flag.ErrHelp
	}
	if cmd.UserName == "" {
		return flag.ErrHelp
	}
	if cmd.Password == "" {
		return flag.ErrHelp
	}
	return nil
}

func (cmd *add) Description() string {
	return `Add host to datacenter.

The host is added to the folder specified by the 'parent' flag. If not given,
this defaults to the hosts folder in the specified or default datacenter.`
}

func (cmd *add) Add(ctx context.Context, parent *object.Folder) error {
	spec := cmd.HostConnectSpec

	req := types.AddStandaloneHost_Task{
		This:         parent.Reference(),
		Spec:         spec,
		AddConnected: cmd.connect,
	}

	res, err := methods.AddStandaloneHost_Task(ctx, parent.Client(), &req)
	if err != nil {
		return err
	}

	logger := cmd.ProgressLogger(fmt.Sprintf("adding %s to folder %s... ", spec.HostName, parent.InventoryPath))
	defer logger.Wait()

	task := object.NewTask(parent.Client(), res.Returnval)
	_, err = task.WaitForResult(ctx, logger)
	return err
}

func (cmd *add) Run(ctx context.Context, f *flag.FlagSet) error {
	var parent *object.Folder

	if f.NArg() != 0 {
		return flag.ErrHelp
	}

	if cmd.parent == "" {
		dc, err := cmd.Datacenter()
		if err != nil {
			return err
		}

		folders, err := dc.Folders(ctx)
		if err != nil {
			return err
		}

		parent = folders.HostFolder
	} else {
		finder, err := cmd.Finder()
		if err != nil {
			return err
		}

		parent, err = finder.Folder(ctx, cmd.parent)
		if err != nil {
			return err
		}
	}

	err := cmd.Add(ctx, parent)
	if err == nil {
		return nil
	}

	// Check if we failed due to SSLVerifyFault and -noverify is set
	if err := cmd.AcceptThumbprint(err); err != nil {
		return err
	}

	// Accepted unverified thumbprint, try again
	return cmd.Add(ctx, parent)
}
