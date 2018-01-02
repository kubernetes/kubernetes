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

package folder

import (
	"context"
	"flag"
	"path"

	"github.com/vmware/govmomi/govc/cli"
	"github.com/vmware/govmomi/govc/flags"
)

type create struct {
	*flags.DatacenterFlag

	pod bool
}

func init() {
	cli.Register("folder.create", &create{})
}

func (cmd *create) Register(ctx context.Context, f *flag.FlagSet) {
	cmd.DatacenterFlag, ctx = flags.NewDatacenterFlag(ctx)
	cmd.DatacenterFlag.Register(ctx, f)

	f.BoolVar(&cmd.pod, "pod", false, "Create folder(s) of type StoragePod (DatastoreCluster)")
}

func (cmd *create) Usage() string {
	return "PATH..."
}

func (cmd *create) Description() string {
	return `Create folder with PATH.

Examples:
  govc folder.create /dc1/vm/folder-foo
  govc object.mv /dc1/vm/vm-foo-* /dc1/vm/folder-foo
  govc folder.create -pod /dc1/datastore/sdrs
  govc object.mv /dc1/datastore/iscsi-* /dc1/datastore/sdrs`
}

func (cmd *create) Process(ctx context.Context) error {
	if err := cmd.DatacenterFlag.Process(ctx); err != nil {
		return err
	}
	return nil
}

func (cmd *create) Run(ctx context.Context, f *flag.FlagSet) error {
	finder, err := cmd.Finder()
	if err != nil {
		return err
	}

	for _, arg := range f.Args() {
		dir := path.Dir(arg)
		name := path.Base(arg)

		if dir == "" {
			dir = "/"
		}

		folder, err := finder.Folder(ctx, dir)
		if err != nil {
			return err
		}

		var create func() error
		if cmd.pod {
			create = func() error {
				_, err = folder.CreateStoragePod(ctx, name)
				return err
			}
		} else {
			create = func() error {
				_, err = folder.CreateFolder(ctx, name)
				return err
			}
		}

		err = create()
		if err != nil {
			return err
		}
	}

	return nil
}
