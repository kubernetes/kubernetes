/*
Copyright (c) 2017 VMware, Inc. All Rights Reserved.

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

package disk

import (
	"context"
	"flag"
	"fmt"

	"github.com/vmware/govmomi/govc/cli"
	"github.com/vmware/govmomi/govc/flags"
	"github.com/vmware/govmomi/object"
)

type inflate struct {
	*flags.DatastoreFlag
}

func init() {
	cli.Register("datastore.disk.inflate", &inflate{})
}

func (cmd *inflate) Register(ctx context.Context, f *flag.FlagSet) {
	cmd.DatastoreFlag, ctx = flags.NewDatastoreFlag(ctx)
	cmd.DatastoreFlag.Register(ctx, f)
}

func (cmd *inflate) Process(ctx context.Context) error {
	return cmd.DatastoreFlag.Process(ctx)
}

func (cmd *inflate) Usage() string {
	return "VMDK"
}

func (cmd *inflate) Description() string {
	return `Inflate VMDK on DS.

Examples:
  govc datastore.disk.inflate disks/disk1.vmdk`
}

func (cmd *inflate) Run(ctx context.Context, f *flag.FlagSet) error {
	dc, err := cmd.Datacenter()
	if err != nil {
		return err
	}

	ds, err := cmd.Datastore()
	if err != nil {
		return err
	}

	m := object.NewVirtualDiskManager(ds.Client())
	path := ds.Path(f.Arg(0))
	task, err := m.InflateVirtualDisk(ctx, path, dc)
	if err != nil {
		return err
	}

	logger := cmd.ProgressLogger(fmt.Sprintf("Inflating %s...", path))
	defer logger.Wait()

	_, err = task.WaitForResult(ctx, logger)
	return err
}
