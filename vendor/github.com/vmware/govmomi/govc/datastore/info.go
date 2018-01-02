/*
Copyright (c) 2015 VMware, Inc. All Rights Reserved.

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
	"io"
	"os"
	"text/tabwriter"

	"github.com/vmware/govmomi/govc/cli"
	"github.com/vmware/govmomi/govc/flags"
	"github.com/vmware/govmomi/object"
	"github.com/vmware/govmomi/property"
	"github.com/vmware/govmomi/vim25/mo"
	"github.com/vmware/govmomi/vim25/types"
)

type info struct {
	*flags.ClientFlag
	*flags.OutputFlag
	*flags.DatacenterFlag
}

func init() {
	cli.Register("datastore.info", &info{})
}

func (cmd *info) Register(ctx context.Context, f *flag.FlagSet) {
	cmd.ClientFlag, ctx = flags.NewClientFlag(ctx)
	cmd.ClientFlag.Register(ctx, f)

	cmd.OutputFlag, ctx = flags.NewOutputFlag(ctx)
	cmd.OutputFlag.Register(ctx, f)

	cmd.DatacenterFlag, ctx = flags.NewDatacenterFlag(ctx)
	cmd.DatacenterFlag.Register(ctx, f)
}

func (cmd *info) Process(ctx context.Context) error {
	if err := cmd.ClientFlag.Process(ctx); err != nil {
		return err
	}
	if err := cmd.OutputFlag.Process(ctx); err != nil {
		return err
	}
	if err := cmd.DatacenterFlag.Process(ctx); err != nil {
		return err
	}
	return nil
}

func (cmd *info) Usage() string {
	return "[PATH]..."
}

func (cmd *info) Run(ctx context.Context, f *flag.FlagSet) error {
	c, err := cmd.Client()
	if err != nil {
		return err
	}

	finder, err := cmd.Finder()
	if err != nil {
		return err
	}

	args := f.Args()
	if len(args) == 0 {
		args = []string{"*"}
	}

	var res infoResult
	var props []string

	if cmd.OutputFlag.JSON {
		props = nil // Load everything
	} else {
		props = []string{"info", "summary"} // Load summary
	}

	for _, arg := range args {
		objects, err := finder.DatastoreList(ctx, arg)
		if err != nil {
			return err
		}
		res.objects = append(res.objects, objects...)
	}

	if len(res.objects) != 0 {
		refs := make([]types.ManagedObjectReference, 0, len(res.objects))
		for _, o := range res.objects {
			refs = append(refs, o.Reference())
		}

		pc := property.DefaultCollector(c)
		err = pc.Retrieve(ctx, refs, props, &res.Datastores)
		if err != nil {
			return err
		}
	}

	return cmd.WriteResult(&res)
}

type infoResult struct {
	Datastores []mo.Datastore
	objects    []*object.Datastore
}

func (r *infoResult) Write(w io.Writer) error {
	// Maintain order via r.objects as Property collector does not always return results in order.
	objects := make(map[types.ManagedObjectReference]mo.Datastore, len(r.Datastores))
	for _, o := range r.Datastores {
		objects[o.Reference()] = o
	}

	tw := tabwriter.NewWriter(os.Stdout, 2, 0, 2, ' ', 0)

	for _, o := range r.objects {
		ds := objects[o.Reference()]
		s := ds.Summary
		fmt.Fprintf(tw, "Name:\t%s\n", s.Name)
		fmt.Fprintf(tw, "  Path:\t%s\n", o.InventoryPath)
		fmt.Fprintf(tw, "  Type:\t%s\n", s.Type)
		fmt.Fprintf(tw, "  URL:\t%s\n", s.Url)
		fmt.Fprintf(tw, "  Capacity:\t%.1f GB\n", float64(s.Capacity)/(1<<30))
		fmt.Fprintf(tw, "  Free:\t%.1f GB\n", float64(s.FreeSpace)/(1<<30))

		switch info := ds.Info.(type) {
		case *types.NasDatastoreInfo:
			fmt.Fprintf(tw, "  Remote:\t%s:%s\n", info.Nas.RemoteHost, info.Nas.RemotePath)
		}
	}

	return tw.Flush()
}
