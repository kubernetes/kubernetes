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
	"os"

	"github.com/vmware/govmomi/govc/cli"
	"github.com/vmware/govmomi/govc/flags"
	"github.com/vmware/govmomi/vim25/soap"
)

type upload struct {
	*flags.OutputFlag
	*flags.DatastoreFlag
}

func init() {
	cli.Register("datastore.upload", &upload{})
}

func (cmd *upload) Register(ctx context.Context, f *flag.FlagSet) {
	cmd.OutputFlag, ctx = flags.NewOutputFlag(ctx)
	cmd.OutputFlag.Register(ctx, f)

	cmd.DatastoreFlag, ctx = flags.NewDatastoreFlag(ctx)
	cmd.DatastoreFlag.Register(ctx, f)
}

func (cmd *upload) Process(ctx context.Context) error {
	if err := cmd.OutputFlag.Process(ctx); err != nil {
		return err
	}
	if err := cmd.DatastoreFlag.Process(ctx); err != nil {
		return err
	}
	return nil
}

func (cmd *upload) Usage() string {
	return "SOURCE DEST"
}

func (cmd *upload) Description() string {
	return `Copy SOURCE from the local system to DEST on DS.

If SOURCE name is "-", read source from stdin.

Examples:
  govc datastore.upload -ds datastore1 ./config.iso vm-name/config.iso
  genisoimage ... | govc datastore.upload -ds datastore1 - vm-name/config.iso`
}

func (cmd *upload) Run(ctx context.Context, f *flag.FlagSet) error {
	args := f.Args()
	if len(args) != 2 {
		return errors.New("invalid arguments")
	}

	ds, err := cmd.Datastore()
	if err != nil {
		return err
	}

	p := soap.DefaultUpload

	src := args[0]
	dst := args[1]

	if src == "-" {
		return ds.Upload(ctx, os.Stdin, dst, &p)
	}

	if cmd.OutputFlag.TTY {
		logger := cmd.ProgressLogger("Uploading... ")
		p.Progress = logger
		defer logger.Wait()
	}

	return ds.UploadFile(ctx, src, dst, &p)
}
