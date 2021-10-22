/*
Copyright (c) 2014-2018 VMware, Inc. All Rights Reserved.

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

package datastore

import (
	"context"
	"flag"
	"fmt"

	"github.com/vmware/govmomi/govc/cli"
	"github.com/vmware/govmomi/govc/flags"
	"github.com/vmware/govmomi/object"
)

type cp struct {
	target
}

func init() {
	cli.Register("datastore.cp", &cp{})
}

type target struct {
	*flags.DatastoreFlag // The source Datastore and the default target Datastore

	dc *flags.DatacenterFlag // Optionally target a different Datacenter
	ds *flags.DatastoreFlag  // Optionally target a different Datastore

	kind  bool
	force bool
}

func (cmd *target) FileManager() (*object.DatastoreFileManager, error) {
	dc, err := cmd.Datacenter()
	if err != nil {
		return nil, err
	}

	ds, err := cmd.Datastore()
	if err != nil {
		return nil, err
	}

	m := ds.NewFileManager(dc, cmd.force)

	dc, err = cmd.dc.Datacenter()
	if err != nil {
		return nil, err
	}
	m.DatacenterTarget = dc

	return m, nil
}

func (cmd *target) Register(ctx context.Context, f *flag.FlagSet) {
	cmd.DatastoreFlag, ctx = flags.NewDatastoreFlag(ctx)
	cmd.DatastoreFlag.Register(ctx, f)

	cmd.dc = &flags.DatacenterFlag{
		OutputFlag: cmd.OutputFlag,
		ClientFlag: cmd.ClientFlag,
	}
	f.StringVar(&cmd.dc.Name, "dc-target", "", "Datacenter destination (defaults to -dc)")

	cmd.ds = &flags.DatastoreFlag{
		DatacenterFlag: cmd.dc,
	}
	f.StringVar(&cmd.ds.Name, "ds-target", "", "Datastore destination (defaults to -ds)")

	f.BoolVar(&cmd.kind, "t", true, "Use file type to choose disk or file manager")
	f.BoolVar(&cmd.force, "f", false, "If true, overwrite any identically named file at the destination")
}

func (cmd *target) Process(ctx context.Context) error {
	if err := cmd.DatastoreFlag.Process(ctx); err != nil {
		return err
	}

	if cmd.dc.Name == "" {
		// Use source DC as target DC
		cmd.dc = cmd.DatacenterFlag
		cmd.ds.DatacenterFlag = cmd.dc
	}

	if cmd.ds.Name == "" {
		// Use source DS as target DS
		cmd.ds.Name = cmd.DatastoreFlag.Name
	}

	return nil
}

func (cmd *cp) Usage() string {
	return "SRC DST"
}

func (cmd *cp) Description() string {
	return `Copy SRC to DST on DATASTORE.

Examples:
  govc datastore.cp foo/foo.vmx foo/foo.vmx.old
  govc datastore.cp -f my.vmx foo/foo.vmx
  govc datastore.cp disks/disk1.vmdk disks/disk2.vmdk
  govc datastore.cp disks/disk1.vmdk -dc-target DC2 disks/disk2.vmdk
  govc datastore.cp disks/disk1.vmdk -ds-target NFS-2 disks/disk2.vmdk`
}

func (cmd *cp) Run(ctx context.Context, f *flag.FlagSet) error {
	args := f.Args()
	if len(args) != 2 {
		return flag.ErrHelp
	}

	m, err := cmd.FileManager()
	if err != nil {
		return err
	}

	src, err := cmd.DatastorePath(args[0])
	if err != nil {
		return err
	}

	dst, err := cmd.target.ds.DatastorePath(args[1])
	if err != nil {
		return err
	}

	cp := m.CopyFile
	if cmd.kind {
		cp = m.Copy
	}

	logger := cmd.ProgressLogger(fmt.Sprintf("Copying %s to %s...", src, dst))
	defer logger.Wait()

	return cp(m.WithProgress(ctx, logger), src, dst)
}
