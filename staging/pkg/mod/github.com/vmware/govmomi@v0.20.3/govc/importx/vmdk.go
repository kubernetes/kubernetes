/*
Copyright (c) 2014-2017 VMware, Inc. All Rights Reserved.

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

package importx

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"path"

	"github.com/vmware/govmomi/govc/cli"
	"github.com/vmware/govmomi/govc/flags"
	"github.com/vmware/govmomi/vmdk"
)

type disk struct {
	*flags.DatastoreFlag
	*flags.ResourcePoolFlag
	*flags.FolderFlag
	*flags.OutputFlag

	force bool
}

func init() {
	cli.Register("import.vmdk", &disk{})
}

func (cmd *disk) Register(ctx context.Context, f *flag.FlagSet) {
	cmd.DatastoreFlag, ctx = flags.NewDatastoreFlag(ctx)
	cmd.DatastoreFlag.Register(ctx, f)
	cmd.ResourcePoolFlag, ctx = flags.NewResourcePoolFlag(ctx)
	cmd.ResourcePoolFlag.Register(ctx, f)
	cmd.FolderFlag, ctx = flags.NewFolderFlag(ctx)
	cmd.FolderFlag.Register(ctx, f)
	cmd.OutputFlag, ctx = flags.NewOutputFlag(ctx)
	cmd.OutputFlag.Register(ctx, f)

	f.BoolVar(&cmd.force, "force", false, "Overwrite existing disk")
}

func (cmd *disk) Process(ctx context.Context) error {
	if err := cmd.DatastoreFlag.Process(ctx); err != nil {
		return err
	}
	if err := cmd.ResourcePoolFlag.Process(ctx); err != nil {
		return err
	}
	if err := cmd.FolderFlag.Process(ctx); err != nil {
		return err
	}
	if err := cmd.OutputFlag.Process(ctx); err != nil {
		return err
	}
	return nil
}

func (cmd *disk) Usage() string {
	return "PATH_TO_VMDK [REMOTE_DIRECTORY]"
}

func (cmd *disk) Run(ctx context.Context, f *flag.FlagSet) error {
	args := f.Args()
	if len(args) < 1 {
		return errors.New("no file to import")
	}

	src := f.Arg(0)

	c, err := cmd.DatastoreFlag.Client()
	if err != nil {
		return err
	}

	dc, err := cmd.DatastoreFlag.Datacenter()
	if err != nil {
		return err
	}

	ds, err := cmd.DatastoreFlag.Datastore()
	if err != nil {
		return err
	}

	pool, err := cmd.ResourcePoolFlag.ResourcePool()
	if err != nil {
		return err
	}

	folder, err := cmd.FolderOrDefault("vm")
	if err != nil {
		return err
	}

	logger := cmd.ProgressLogger(fmt.Sprintf("Uploading %s... ", path.Base(src)))
	defer logger.Wait()

	p := vmdk.ImportParams{
		Path:       f.Arg(1),
		Logger:     logger,
		Type:       "", // TODO: flag
		Force:      cmd.force,
		Datacenter: dc,
		Pool:       pool,
		Folder:     folder,
	}

	err = vmdk.Import(ctx, c, src, ds, p)
	if err != nil && err == vmdk.ErrInvalidFormat {
		return fmt.Errorf(`%s
The vmdk can be converted using one of:
  vmware-vdiskmanager -t 5 -r '%s' new.vmdk
  qemu-img convert -O vmdk -o subformat=streamOptimized '%s' new.vmdk`, err, src, src)
	}

	return err
}
