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
	"io"
	"strings"
	"text/tabwriter"
	"time"

	"github.com/vmware/govmomi/govc/cli"
	"github.com/vmware/govmomi/govc/flags"
	"github.com/vmware/govmomi/units"
	"github.com/vmware/govmomi/vim25/types"
	"github.com/vmware/govmomi/vslm"
)

type ls struct {
	*flags.DatastoreFlag
	long     bool
	category string
	tag      string
	tags     bool
}

func init() {
	cli.Register("disk.ls", &ls{})
}

func (cmd *ls) Register(ctx context.Context, f *flag.FlagSet) {
	cmd.DatastoreFlag, ctx = flags.NewDatastoreFlag(ctx)
	cmd.DatastoreFlag.Register(ctx, f)

	f.BoolVar(&cmd.long, "l", false, "Long listing format")
	f.StringVar(&cmd.category, "c", "", "Query tag category")
	f.StringVar(&cmd.tag, "t", "", "Query tag name")
	f.BoolVar(&cmd.tags, "T", false, "List attached tags")
}

func (cmd *ls) Usage() string {
	return "[ID]..."
}

func (cmd *ls) Description() string {
	return `List disk IDs on DS.

Examples:
  govc disk.ls
  govc disk.ls -l -T
  govc disk.ls -l e9b06a8b-d047-4d3c-b15b-43ea9608b1a6
  govc disk.ls -c k8s-region -t us-west-2`
}

type VStorageObject struct {
	types.VStorageObject
	Tags []types.VslmTagEntry
}

func (o *VStorageObject) tags() string {
	var tags []string
	for _, tag := range o.Tags {
		tags = append(tags, tag.ParentCategoryName+":"+tag.TagName)
	}
	return strings.Join(tags, ",")
}

type lsResult struct {
	cmd     *ls
	Objects []VStorageObject
}

func (r *lsResult) Write(w io.Writer) error {
	tw := tabwriter.NewWriter(r.cmd.Out, 2, 0, 2, ' ', 0)

	for _, o := range r.Objects {
		_, _ = fmt.Fprintf(tw, "%s\t%s", o.Config.Id.Id, o.Config.Name)
		if r.cmd.long {
			created := o.Config.CreateTime.Format(time.Stamp)
			size := units.FileSize(o.Config.CapacityInMB * 1024 * 1024)
			_, _ = fmt.Fprintf(tw, "\t%s\t%s", size, created)
		}
		if r.cmd.tags {
			_, _ = fmt.Fprintf(tw, "\t%s", o.tags())
		}
		_, _ = fmt.Fprintln(tw)
	}

	return tw.Flush()
}

func (r *lsResult) Dump() interface{} {
	return r.Objects
}

func (cmd *ls) Run(ctx context.Context, f *flag.FlagSet) error {
	c, err := cmd.Client()
	if err != nil {
		return err
	}

	ds, err := cmd.Datastore()
	if err != nil {
		return err
	}

	m := vslm.NewObjectManager(c)
	res := lsResult{cmd: cmd}

	ids := f.Args()
	if len(ids) == 0 {
		var oids []types.ID
		if cmd.category == "" {
			oids, err = m.List(ctx, ds)
		} else {
			oids, err = m.ListAttachedObjects(ctx, cmd.category, cmd.tag)
		}

		if err != nil {
			return err
		}
		for _, id := range oids {
			ids = append(ids, id.Id)
		}
	}

	for _, id := range ids {
		o, err := m.Retrieve(ctx, ds, id)
		if err != nil {
			return err
		}

		obj := VStorageObject{VStorageObject: *o}
		if cmd.tags {
			obj.Tags, err = m.ListAttachedTags(ctx, id)
			if err != nil {
				return err
			}
		}
		res.Objects = append(res.Objects, obj)
	}

	return cmd.WriteResult(&res)
}
