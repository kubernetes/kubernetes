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

package object

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"reflect"
	"strings"
	"text/tabwriter"

	"github.com/vmware/govmomi/govc/cli"
	"github.com/vmware/govmomi/govc/flags"
	"github.com/vmware/govmomi/property"
	"github.com/vmware/govmomi/vim25/methods"
	"github.com/vmware/govmomi/vim25/types"
)

type collect struct {
	*flags.DatacenterFlag

	single bool
	simple bool
	n      int
}

func init() {
	cli.Register("object.collect", &collect{})
}

func (cmd *collect) Register(ctx context.Context, f *flag.FlagSet) {
	cmd.DatacenterFlag, ctx = flags.NewDatacenterFlag(ctx)
	cmd.DatacenterFlag.Register(ctx, f)

	f.BoolVar(&cmd.simple, "s", false, "Output property value only")
	f.IntVar(&cmd.n, "n", 0, "Wait for N property updates")
}

func (cmd *collect) Usage() string {
	return "[MOID] [PROPERTY]..."
}

func (cmd *collect) Description() string {
	return `Collect managed object properties.

MOID can be an inventory path or ManagedObjectReference.
MOID defaults to '-', an alias for 'ServiceInstance:ServiceInstance'.

By default only the current property value(s) are collected.  Use the '-n' flag to wait for updates.

Examples:
  govc object.collect - content
  govc object.collect -s HostSystem:ha-host hardware.systemInfo.uuid
  govc object.collect -s /ha-datacenter/vm/foo overallStatus
  govc object.collect -json -n=-1 EventManager:ha-eventmgr latestEvent | jq .
  govc object.collect -json -s $(govc object.collect -s - content.perfManager) description.counterType | jq .`
}

func (cmd *collect) Process(ctx context.Context) error {
	if err := cmd.DatacenterFlag.Process(ctx); err != nil {
		return err
	}
	return nil
}

var stringer = reflect.TypeOf((*fmt.Stringer)(nil)).Elem()

type change struct {
	cmd            *collect
	PropertyChange []types.PropertyChange
}

func (pc *change) MarshalJSON() ([]byte, error) {
	return json.Marshal(pc.PropertyChange)
}

func (pc *change) output(name string, rval reflect.Value, rtype reflect.Type) {
	s := "..."

	kind := rval.Kind()

	if kind == reflect.Ptr || kind == reflect.Interface {
		if rval.IsNil() {
			s = ""
		} else {
			rval = rval.Elem()
			kind = rval.Kind()
		}
	}

	switch kind {
	case reflect.Ptr, reflect.Interface:
	case reflect.Slice:
		if rval.Len() == 0 {
			s = ""
			break
		}

		etype := rtype.Elem()

		if etype.Kind() != reflect.Interface && etype.Kind() != reflect.Struct || etype.Implements(stringer) {
			var val []string

			for i := 0; i < rval.Len(); i++ {
				v := rval.Index(i).Interface()

				if fstr, ok := v.(fmt.Stringer); ok {
					s = fstr.String()
				} else {
					s = fmt.Sprintf("%v", v)
				}

				val = append(val, s)
			}

			s = strings.Join(val, ",")
		}
	case reflect.Struct:
		if rtype.Implements(stringer) {
			s = rval.Interface().(fmt.Stringer).String()
		}
	default:
		s = fmt.Sprintf("%v", rval.Interface())
	}

	if pc.cmd.simple {
		fmt.Fprintln(pc.cmd.Out, s)
		return
	}

	fmt.Fprintf(pc.cmd.Out, "%s\t%s\t%s\n", name, rtype, s)
}

func (pc *change) writeStruct(name string, rval reflect.Value, rtype reflect.Type) {
	for i := 0; i < rval.NumField(); i++ {
		fval := rval.Field(i)
		field := rtype.Field(i)

		if field.Anonymous {
			pc.writeStruct(name, fval, fval.Type())
			continue
		}

		fname := fmt.Sprintf("%s.%s%s", name, strings.ToLower(field.Name[:1]), field.Name[1:])
		pc.output(fname, fval, field.Type)
	}
}

func (pc *change) Write(w io.Writer) error {
	tw := tabwriter.NewWriter(pc.cmd.Out, 4, 0, 2, ' ', 0)
	pc.cmd.Out = tw

	for _, c := range pc.PropertyChange {
		if c.Val == nil {
			// type is unknown in this case, as xsi:type was not provided - just skip for now
			continue
		}

		rval := reflect.ValueOf(c.Val)
		rtype := rval.Type()

		if strings.HasPrefix(rtype.Name(), "ArrayOf") {
			rval = rval.Field(0)
			rtype = rval.Type()
		}

		if pc.cmd.single && rtype.Kind() == reflect.Struct && !rtype.Implements(stringer) {
			pc.writeStruct(c.Name, rval, rtype)
			continue
		}

		pc.output(c.Name, rval, rtype)
	}

	return tw.Flush()
}

func (cmd *collect) Run(ctx context.Context, f *flag.FlagSet) error {
	client, err := cmd.Client()
	if err != nil {
		return err
	}

	finder, err := cmd.Finder()
	if err != nil {
		return err
	}

	ref := methods.ServiceInstance
	arg := f.Arg(0)

	switch arg {
	case "", "-":
	default:
		if !ref.FromString(arg) {
			l, ferr := finder.ManagedObjectList(ctx, arg)
			if ferr != nil {
				return err
			}

			switch len(l) {
			case 0:
				return fmt.Errorf("%s not found", arg)
			case 1:
				ref = l[0].Object.Reference()
			default:
				return flag.ErrHelp
			}
		}
	}

	p := property.DefaultCollector(client)

	var props []string
	if f.NArg() > 1 {
		props = f.Args()[1:]
		cmd.single = len(props) == 1
	}

	return property.Wait(ctx, p, ref, props, func(pc []types.PropertyChange) bool {
		_ = cmd.WriteResult(&change{cmd, pc})

		cmd.n--

		return cmd.n == -1
	})
}
