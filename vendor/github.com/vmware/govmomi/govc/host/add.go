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
	"context"
	"flag"
	"fmt"

	"github.com/vmware/govmomi/govc/cli"
	"github.com/vmware/govmomi/govc/flags"
	"github.com/vmware/govmomi/object"
	"github.com/vmware/govmomi/vim25/methods"
	"github.com/vmware/govmomi/vim25/types"
)

type add struct {
	*flags.FolderFlag
	*flags.HostConnectFlag

	connect bool
}

func init() {
	cli.Register("host.add", &add{})
}

func (cmd *add) Register(ctx context.Context, f *flag.FlagSet) {
	cmd.FolderFlag, ctx = flags.NewFolderFlag(ctx)
	cmd.FolderFlag.Register(ctx, f)

	cmd.HostConnectFlag, ctx = flags.NewHostConnectFlag(ctx)
	cmd.HostConnectFlag.Register(ctx, f)

	f.BoolVar(&cmd.connect, "connect", true, "Immediately connect to host")
}

func (cmd *add) Process(ctx context.Context) error {
	if err := cmd.FolderFlag.Process(ctx); err != nil {
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

The host is added to the folder specified by the 'folder' flag. If not given,
this defaults to the host folder in the specified or default datacenter.

Examples:
  thumbprint=$(govc about.cert -k -u host.example.com -thumbprint | awk '{print $2}')
  govc host.add -hostname host.example.com -username root -password pass -thumbprint $thumbprint
  govc host.add -hostname 10.0.6.1 -username root -password pass -noverify`
}

func (cmd *add) Add(ctx context.Context, parent *object.Folder) error {
	spec := cmd.Spec(parent.Client())

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
	if f.NArg() != 0 {
		return flag.ErrHelp
	}

	folder, err := cmd.FolderOrDefault("host")
	if err != nil {
		return err
	}

	return cmd.Fault(cmd.Add(ctx, folder))
}
