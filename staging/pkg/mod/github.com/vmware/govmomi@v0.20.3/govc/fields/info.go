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

package fields

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
	*flags.DatacenterFlag

	name string
}

func init() {
	cli.Register("fields.info", &info{})
}

func (cmd *info) Register(ctx context.Context, f *flag.FlagSet) {
	cmd.DatacenterFlag, ctx = flags.NewDatacenterFlag(ctx)
	cmd.DatacenterFlag.Register(ctx, f)

	f.StringVar(&cmd.name, "n", "", "Filter by custom field name")
}

func (cmd *info) Usage() string {
	return "PATH..."
}

func (cmd *info) Description() string {
	return `Display custom field values for PATH.

Also known as "Custom Attributes".

Examples:
  govc fields.info vm/*
  govc fields.info -n my-field-name vm/*`
}

type Info struct {
	Object types.ManagedObjectReference
	Path   string
	Name   string
	Key    string
	Value  string
}

type infoResult struct {
	Info []Info
}

func (r *infoResult) Write(w io.Writer) error {
	tw := tabwriter.NewWriter(os.Stdout, 2, 0, 2, ' ', 0)

	for _, info := range r.Info {
		_, _ = fmt.Fprintf(tw, "%s\t%s\t%s\n", info.Path, info.Key, info.Value)
	}

	return tw.Flush()
}

func (cmd *info) Run(ctx context.Context, f *flag.FlagSet) error {
	c, err := cmd.Client()
	if err != nil {
		return err
	}

	m, err := object.GetCustomFieldsManager(c)
	if err != nil {
		return err
	}

	paths := make(map[types.ManagedObjectReference]string)
	var refs []types.ManagedObjectReference

	finder, err := cmd.Finder()
	if err != nil {
		return err
	}

	for _, arg := range f.Args() {
		elements, ferr := finder.ManagedObjectList(ctx, arg)
		if ferr != nil {
			return ferr
		}

		if len(elements) == 0 {
			return fmt.Errorf("object '%s' not found", arg)
		}

		for _, e := range elements {
			ref := e.Object.Reference()
			refs = append(refs, ref)
			paths[ref] = e.Path
		}
	}

	var objs []mo.ManagedEntity
	err = property.DefaultCollector(c).Retrieve(ctx, refs, []string{"name", "customValue"}, &objs)
	if err != nil {
		return err
	}

	matches := func(key int32) bool {
		return true
	}

	if cmd.name != "" {
		fkey, cerr := m.FindKey(ctx, cmd.name)
		if cerr != nil {
			return err
		}
		matches = func(key int32) bool {
			return key == fkey
		}
	}

	field, err := m.Field(ctx)
	if err != nil {
		return err
	}

	var res infoResult

	for _, obj := range objs {
		for i := range obj.CustomValue {
			val := obj.CustomValue[i].(*types.CustomFieldStringValue)

			if !matches(val.Key) {
				continue
			}

			res.Info = append(res.Info, Info{
				Object: obj.Self,
				Path:   paths[obj.Self],
				Name:   obj.Name,
				Key:    field.ByKey(val.Key).Name,
				Value:  val.Value,
			})
		}
	}

	return cmd.WriteResult(&res)
}
