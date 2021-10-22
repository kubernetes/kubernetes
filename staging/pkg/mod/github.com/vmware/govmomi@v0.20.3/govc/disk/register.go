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

type register struct {
	*flags.DatastoreFlag
}

func init() {
	cli.Register("disk.register", &register{})
}

func (cmd *register) Register(ctx context.Context, f *flag.FlagSet) {
	cmd.DatastoreFlag, ctx = flags.NewDatastoreFlag(ctx)
	cmd.DatastoreFlag.Register(ctx, f)
}

func (cmd *register) Usage() string {
	return "PATH [NAME]"
}

func (cmd *register) Description() string {
	return `Register existing disk on DS.

Examples:
  govc disk.register disks/disk1.vmdk my-disk`
}

func (cmd *register) Run(ctx context.Context, f *flag.FlagSet) error {
	ds, err := cmd.Datastore()
	if err != nil {
		return err
	}

	m := vslm.NewObjectManager(ds.Client())

	path := ds.NewURL(f.Arg(0)).String()

	obj, err := m.RegisterDisk(ctx, path, f.Arg(1))
	if err != nil {
		return err
	}

	fmt.Println(obj.Config.Id.Id)

	return nil
}
