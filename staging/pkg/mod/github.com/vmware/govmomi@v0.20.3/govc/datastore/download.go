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
	"fmt"
	"io"
	"os"

	"github.com/vmware/govmomi/govc/cli"
	"github.com/vmware/govmomi/govc/flags"
	"github.com/vmware/govmomi/vim25/soap"
)

type download struct {
	*flags.DatastoreFlag
	*flags.HostSystemFlag
}

func init() {
	cli.Register("datastore.download", &download{})
}

func (cmd *download) Register(ctx context.Context, f *flag.FlagSet) {
	cmd.DatastoreFlag, ctx = flags.NewDatastoreFlag(ctx)
	cmd.DatastoreFlag.Register(ctx, f)

	cmd.HostSystemFlag, ctx = flags.NewHostSystemFlag(ctx)
	cmd.HostSystemFlag.Register(ctx, f)
}

func (cmd *download) Process(ctx context.Context) error {
	if err := cmd.DatastoreFlag.Process(ctx); err != nil {
		return err
	}
	if err := cmd.HostSystemFlag.Process(ctx); err != nil {
		return err
	}
	return nil
}

func (cmd *download) Usage() string {
	return "SOURCE DEST"
}

func (cmd *download) Description() string {
	return `Copy SOURCE from DS to DEST on the local system.

If DEST name is "-", source is written to stdout.

Examples:
  govc datastore.download vm-name/vmware.log ./local.log
  govc datastore.download vm-name/vmware.log - | grep -i error`
}

func (cmd *download) Run(ctx context.Context, f *flag.FlagSet) error {
	args := f.Args()
	if len(args) != 2 {
		return errors.New("invalid arguments")
	}

	ds, err := cmd.Datastore()
	if err != nil {
		return err
	}

	h, err := cmd.HostSystemIfSpecified()
	if err != nil {
		return err
	}

	var via string

	if h != nil {
		via = fmt.Sprintf(" via %s", h.InventoryPath)
		ctx = ds.HostContext(ctx, h)
	}

	p := soap.DefaultDownload

	src := args[0]
	dst := args[1]

	if dst == "-" {
		f, _, err := ds.Download(ctx, src, &p)
		if err != nil {
			return err
		}
		_, err = io.Copy(os.Stdout, f)
		return err
	}

	if cmd.DatastoreFlag.OutputFlag.TTY {
		logger := cmd.DatastoreFlag.ProgressLogger(fmt.Sprintf("Downloading%s... ", via))
		p.Progress = logger
		defer logger.Wait()
	}

	return ds.DownloadFile(ctx, src, dst, &p)
}
