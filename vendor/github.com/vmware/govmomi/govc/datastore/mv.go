/*
Copyright (c) 2014-2015 VMware, Inc. All Rights Reserved.

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
	"errors"
	"flag"

	"github.com/vmware/govmomi/govc/cli"
	"github.com/vmware/govmomi/govc/flags"
	"github.com/vmware/govmomi/object"
)

type mv struct {
	*flags.DatastoreFlag

	force bool
}

func init() {
	cli.Register("datastore.mv", &mv{})
}

func (cmd *mv) Register(ctx context.Context, f *flag.FlagSet) {
	cmd.DatastoreFlag, ctx = flags.NewDatastoreFlag(ctx)
	cmd.DatastoreFlag.Register(ctx, f)

	f.BoolVar(&cmd.force, "f", false, "If true, overwrite any identically named file at the destination")
}

func (cmd *mv) Process(ctx context.Context) error {
	if err := cmd.DatastoreFlag.Process(ctx); err != nil {
		return err
	}
	return nil
}

func (cmd *mv) Usage() string {
	return "SRC DST"
}

func (cmd *mv) Description() string {
	return `Move SRC to DST on DATASTORE.

Examples:
  govc datastore.mv foo/foo.vmx foo/foo.vmx.old
  govc datastore.mv -f my.vmx foo/foo.vmx`
}

func (cmd *mv) Run(ctx context.Context, f *flag.FlagSet) error {
	args := f.Args()
	if len(args) != 2 {
		return errors.New("SRC and DST arguments are required")
	}

	c, err := cmd.Client()
	if err != nil {
		return err
	}

	dc, err := cmd.Datacenter()
	if err != nil {
		return err
	}

	// TODO: support cross-datacenter move

	src, err := cmd.DatastorePath(args[0])
	if err != nil {
		return err
	}

	dst, err := cmd.DatastorePath(args[1])
	if err != nil {
		return err
	}

	m := object.NewFileManager(c)
	task, err := m.MoveDatastoreFile(ctx, src, dc, dst, dc, cmd.force)
	if err != nil {
		return err
	}

	return task.Wait(ctx)
}
