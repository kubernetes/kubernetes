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
	"io"
	"os"
	"text/tabwriter"
	"time"

	"github.com/vmware/govmomi/govc/cli"
	"github.com/vmware/govmomi/govc/flags"
	"github.com/vmware/govmomi/vim25/types"
	"github.com/vmware/govmomi/vslm"
)

type ls struct {
	*flags.DatastoreFlag
	long bool
}

func init() {
	cli.Register("disk.snapshot.ls", &ls{})
}

func (cmd *ls) Register(ctx context.Context, f *flag.FlagSet) {
	cmd.DatastoreFlag, ctx = flags.NewDatastoreFlag(ctx)
	cmd.DatastoreFlag.Register(ctx, f)

	f.BoolVar(&cmd.long, "l", false, "Long listing format")
}

func (cmd *ls) Usage() string {
	return "ID"
}

func (cmd *ls) Description() string {
	return `List snapshots for disk ID on DS.

Examples:
  govc snapshot.disk.ls -l 9b06a8b-d047-4d3c-b15b-43ea9608b1a6`
}

type lsResult struct {
	Info *types.VStorageObjectSnapshotInfo
	cmd  *ls
}

func (r *lsResult) Write(w io.Writer) error {
	tw := tabwriter.NewWriter(os.Stdout, 2, 0, 2, ' ', 0)

	for _, o := range r.Info.Snapshots {
		_, _ = fmt.Fprintf(tw, "%s\t%s", o.Id.Id, o.Description)
		if r.cmd.long {
			created := o.CreateTime.Format(time.Stamp)
			_, _ = fmt.Fprintf(tw, "\t%s", created)
		}
		_, _ = fmt.Fprintln(tw)
	}

	return tw.Flush()
}

func (r *lsResult) Dump() interface{} {
	return r.Info
}

func (cmd *ls) Run(ctx context.Context, f *flag.FlagSet) error {
	ds, err := cmd.Datastore()
	if err != nil {
		return err
	}

	m := vslm.NewObjectManager(ds.Client())
	info, err := m.RetrieveSnapshotInfo(ctx, ds, f.Arg(0))
	if err != nil {
		return err
	}

	res := lsResult{Info: info, cmd: cmd}

	return cmd.WriteResult(&res)
}
