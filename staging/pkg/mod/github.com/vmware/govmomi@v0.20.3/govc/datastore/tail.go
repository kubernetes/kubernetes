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

package datastore

import (
	"context"
	"flag"
	"io"
	"os"
	"time"

	"github.com/vmware/govmomi/govc/cli"
	"github.com/vmware/govmomi/govc/flags"
)

type tail struct {
	*flags.DatastoreFlag
	*flags.HostSystemFlag

	count  int64
	lines  int
	follow bool
}

func init() {
	cli.Register("datastore.tail", &tail{})
}

func (cmd *tail) Register(ctx context.Context, f *flag.FlagSet) {
	cmd.DatastoreFlag, ctx = flags.NewDatastoreFlag(ctx)
	cmd.DatastoreFlag.Register(ctx, f)

	cmd.HostSystemFlag, ctx = flags.NewHostSystemFlag(ctx)
	cmd.HostSystemFlag.Register(ctx, f)

	f.Int64Var(&cmd.count, "c", -1, "Output the last NUM bytes")
	f.IntVar(&cmd.lines, "n", 10, "Output the last NUM lines")
	f.BoolVar(&cmd.follow, "f", false, "Output appended data as the file grows")
}

func (cmd *tail) Description() string {
	return `Output the last part of datastore files.

Examples:
  govc datastore.tail -n 100 vm-name/vmware.log
  govc datastore.tail -n 0 -f vm-name/vmware.log`
}

func (cmd *tail) Process(ctx context.Context) error {
	if err := cmd.DatastoreFlag.Process(ctx); err != nil {
		return err
	}
	if err := cmd.HostSystemFlag.Process(ctx); err != nil {
		return err
	}
	return nil
}

func (cmd *tail) Usage() string {
	return "PATH"
}

func (cmd *tail) Run(ctx context.Context, f *flag.FlagSet) error {
	if f.NArg() != 1 {
		return flag.ErrHelp
	}

	p := cmd.Args(f.Args())[0]

	ds, err := cmd.Datastore()
	if err != nil {
		return err
	}

	h, err := cmd.HostSystemIfSpecified()
	if err != nil {
		return err
	}

	if h != nil {
		ctx = ds.HostContext(ctx, h)
	}

	file, err := ds.Open(ctx, p.Path)
	if err != nil {
		return err
	}

	var reader io.ReadCloser = file

	var offset int64

	if cmd.count >= 0 {
		info, serr := file.Stat()
		if serr != nil {
			return serr
		}

		if info.Size() > cmd.count {
			offset = info.Size() - cmd.count

			_, err = file.Seek(offset, io.SeekStart)
			if err != nil {
				return err
			}
		}
	} else if cmd.lines >= 0 {
		err = file.Tail(cmd.lines)
		if err != nil {
			return err
		}
	}

	if cmd.follow {
		reader = file.Follow(time.Second)
	}

	_, err = io.Copy(os.Stdout, reader)

	_ = reader.Close()

	return err
}
