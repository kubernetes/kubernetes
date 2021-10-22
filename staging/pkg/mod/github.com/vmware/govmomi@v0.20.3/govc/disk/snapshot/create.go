/*
Copyright (c) 2018 VMware, Inc. All Rights Reserved.

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

package snapshot

import (
	"context"
	"flag"
	"fmt"

	"github.com/vmware/govmomi/govc/cli"
	"github.com/vmware/govmomi/govc/flags"
	"github.com/vmware/govmomi/vim25/types"
	"github.com/vmware/govmomi/vslm"
)

type create struct {
	*flags.DatastoreFlag
}

func init() {
	cli.Register("disk.snapshot.create", &create{})
}

func (cmd *create) Register(ctx context.Context, f *flag.FlagSet) {
	cmd.DatastoreFlag, ctx = flags.NewDatastoreFlag(ctx)
	cmd.DatastoreFlag.Register(ctx, f)
}

func (cmd *create) Usage() string {
	return "ID DESC"
}

func (cmd *create) Description() string {
	return `Create snapshot of ID on DS.

Examples:
  govc disk.snapshot.create b9fe5f17-3b87-4a03-9739-09a82ddcc6b0 my-disk-snapshot`
}

func (cmd *create) Run(ctx context.Context, f *flag.FlagSet) error {
	ds, err := cmd.Datastore()
	if err != nil {
		return err
	}

	m := vslm.NewObjectManager(ds.Client())
	id := f.Arg(0)

	task, err := m.CreateSnapshot(ctx, ds, id, f.Arg(1))
	if err != nil {
		return err
	}

	logger := cmd.ProgressLogger(fmt.Sprintf("Snapshot %s...", id))

	res, err := task.WaitForResult(ctx, logger)
	logger.Wait()
	if err != nil {
		return err
	}

	fmt.Println(res.Result.(types.ID).Id)

	return nil
}
