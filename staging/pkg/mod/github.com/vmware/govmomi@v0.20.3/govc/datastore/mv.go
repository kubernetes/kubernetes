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
)

type mv struct {
	target
}

func init() {
	cli.Register("datastore.mv", &mv{})
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

	mv := m.MoveFile
	if cmd.kind {
		mv = m.Move
	}

	logger := cmd.ProgressLogger(fmt.Sprintf("Moving %s to %s...", src, dst))
	defer logger.Wait()

	return mv(m.WithProgress(ctx, logger), src, dst)
}
