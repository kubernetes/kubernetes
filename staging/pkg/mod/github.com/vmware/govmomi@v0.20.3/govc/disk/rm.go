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

package disk

import (
	"context"
	"flag"
	"fmt"

	"github.com/vmware/govmomi/govc/cli"
	"github.com/vmware/govmomi/govc/flags"
	"github.com/vmware/govmomi/vslm"
)

type rm struct {
	*flags.DatastoreFlag
}

func init() {
	cli.Register("disk.rm", &rm{})
}

func (cmd *rm) Register(ctx context.Context, f *flag.FlagSet) {
	cmd.DatastoreFlag, ctx = flags.NewDatastoreFlag(ctx)
	cmd.DatastoreFlag.Register(ctx, f)
}

func (cmd *rm) Usage() string {
	return "ID"
}

func (cmd *rm) Description() string {
	return `Remove disk ID on DS.

Examples:
  govc disk.rm ID`
}

func (cmd *rm) Run(ctx context.Context, f *flag.FlagSet) error {
	ds, err := cmd.Datastore()
	if err != nil {
		return err
	}

	id := f.Arg(0)
	m := vslm.NewObjectManager(ds.Client())

	task, err := m.Delete(ctx, ds, id)

	logger := cmd.ProgressLogger(fmt.Sprintf("Deleting %s...", id))
	defer logger.Wait()

	_, err = task.WaitForResult(ctx, logger)
	return err
}
