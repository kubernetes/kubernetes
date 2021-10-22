/*
Copyright (c) 2017 VMware, Inc. All Rights Reserved.

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

package vsan

import (
	"context"
	"flag"
	"fmt"
	"net/url"
	"text/tabwriter"

	"github.com/vmware/govmomi/govc/cli"
	"github.com/vmware/govmomi/govc/flags"
	"github.com/vmware/govmomi/object"
	"github.com/vmware/govmomi/vim25/mo"
)

type ls struct {
	*flags.DatastoreFlag

	long   bool
	orphan bool
}

func init() {
	cli.Register("datastore.vsan.dom.ls", &ls{})
}

func (cmd *ls) Register(ctx context.Context, f *flag.FlagSet) {
	cmd.DatastoreFlag, ctx = flags.NewDatastoreFlag(ctx)
	cmd.DatastoreFlag.Register(ctx, f)

	f.BoolVar(&cmd.long, "l", false, "Long listing")
	f.BoolVar(&cmd.orphan, "o", false, "List orphan objects")
}

func (cmd *ls) Process(ctx context.Context) error {
	if err := cmd.DatastoreFlag.Process(ctx); err != nil {
		return err
	}
	return nil
}

func (cmd *ls) Usage() string {
	return "[UUID]..."
}

func (cmd *ls) Description() string {
	return `List vSAN DOM objects in DS.

Examples:
  govc datastore.vsan.dom.ls
  govc datastore.vsan.dom.ls -ds vsanDatastore -l
  govc datastore.vsan.dom.ls -l d85aa758-63f5-500a-3150-0200308e589c`
}

func (cmd *ls) Run(ctx context.Context, f *flag.FlagSet) error {
	ds, err := cmd.Datastore()
	if err != nil {
		return err
	}

	var mds mo.Datastore
	err = ds.Properties(ctx, ds.Reference(), []string{"summary"}, &mds)
	if err != nil {
		return err
	}

	if mds.Summary.Type != "vsan" {
		return flag.ErrHelp
	}

	hosts, err := ds.AttachedHosts(ctx)
	if err != nil {
		return err
	}

	if len(hosts) == 0 {
		return flag.ErrHelp
	}

	m, err := hosts[0].ConfigManager().VsanInternalSystem(ctx)
	if err != nil {
		return err
	}

	ids, err := m.QueryVsanObjectUuidsByFilter(ctx, f.Args(), 0, 0)
	if err != nil {
		return err
	}

	if len(ids) == 0 {
		return nil
	}

	if !cmd.long && !cmd.orphan {
		for _, id := range ids {
			fmt.Fprintln(cmd.Out, id)
		}

		return nil
	}

	objs, err := m.GetVsanObjExtAttrs(ctx, ids)
	if err != nil {
		return err
	}

	u, err := url.Parse(mds.Summary.Url)
	if err != nil {
		return err
	}

	tw := tabwriter.NewWriter(cmd.Out, 2, 0, 2, ' ', 0)
	cmd.Out = tw

	for id, obj := range objs {
		path := obj.DatastorePath(u.Path)

		if cmd.orphan {
			_, err = ds.Stat(ctx, path)
			if err == nil {
				continue
			}

			switch err.(type) {
			case object.DatastoreNoSuchDirectoryError, object.DatastoreNoSuchFileError:
			default:
				return err
			}

			if !cmd.long {
				fmt.Fprintln(cmd.Out, id)
				continue
			}
		}

		fmt.Fprintf(cmd.Out, "%s\t%s\t%s\n", id, obj.Class, path)
	}

	return tw.Flush()
}
