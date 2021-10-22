/*
Copyright (c) 2016-2017 VMware, Inc. All Rights Reserved.

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
	"os"
	"reflect"
	"strings"
	"text/tabwriter"
	"time"

	"github.com/vmware/govmomi/govc/cli"
	"github.com/vmware/govmomi/govc/flags"
	"github.com/vmware/govmomi/property"
	"github.com/vmware/govmomi/view"
	"github.com/vmware/govmomi/vim25"
	"github.com/vmware/govmomi/vim25/methods"
	"github.com/vmware/govmomi/vim25/soap"
	"github.com/vmware/govmomi/vim25/types"
	"github.com/vmware/govmomi/vim25/xml"
)

type collect struct {
	*flags.DatacenterFlag

	single bool
	simple bool
	raw    string
	dump   bool
	n      int
	kind   kinds
	wait   time.Duration

	filter property.Filter
	obj    string
}

func init() {
	cli.Register("object.collect", &collect{})
}

func (cmd *collect) Register(ctx context.Context, f *flag.FlagSet) {
	cmd.DatacenterFlag, ctx = flags.NewDatacenterFlag(ctx)
	cmd.DatacenterFlag.Register(ctx, f)

	f.BoolVar(&cmd.simple, "s", false, "Output property value only")
	f.BoolVar(&cmd.dump, "O", false, "Output the CreateFilter request itself")
	f.StringVar(&cmd.raw, "R", "", "Raw XML encoded CreateFilter request")
	f.IntVar(&cmd.n, "n", 0, "Wait for N property updates")
	f.Var(&cmd.kind, "type", "Resource type.  If specified, MOID is used for a container view root")
	f.DurationVar(&cmd.wait, "wait", 0, "Max wait time for updates")
}

func (cmd *collect) Usage() string {
	return "[MOID] [PROPERTY]..."
}

func (cmd *collect) Description() string {
	return `Collect managed object properties.

MOID can be an inventory path or ManagedObjectReference.
MOID defaults to '-', an alias for 'ServiceInstance:ServiceInstance' or the root folder if a '-type' flag is given.

If a '-type' flag is given, properties are collected using a ContainerView object where MOID is the root of the view.

By default only the current property value(s) are collected.  To wait for updates, use the '-n' flag or
specify a property filter.  A property filter can be specified by prefixing the property name with a '-',
followed by the value to match.

The '-R' flag sets the Filter using the given XML encoded request, which can be captured by 'vcsim -trace' for example.
It can be useful for replaying property filters created by other clients and converting filters to Go code via '-O -dump'.

Examples:
  govc object.collect - content
  govc object.collect -s HostSystem:ha-host hardware.systemInfo.uuid
  govc object.collect -s /ha-datacenter/vm/foo overallStatus
  govc object.collect -s /ha-datacenter/vm/foo -guest.guestOperationsReady true # property filter
  govc object.collect -type m / name runtime.powerState # collect properties for multiple objects
  govc object.collect -json -n=-1 EventManager:ha-eventmgr latestEvent | jq .
  govc object.collect -json -s $(govc object.collect -s - content.perfManager) description.counterType | jq .
  govc object.collect -R create-filter-request.xml # replay filter
  govc object.collect -R create-filter-request.xml -O # convert filter to Go code
  govc object.collect -s vm/my-vm summary.runtime.host | xargs govc ls -L # inventory path of VM's host
  govc object.collect -json $vm config | \ # use -json + jq to search array elements
    jq -r '.[] | select(.Val.Hardware.Device[].MacAddress == "00:0c:29:0c:73:c0") | .Val.Name'`
}

var stringer = reflect.TypeOf((*fmt.Stringer)(nil)).Elem()

type change struct {
	cmd    *collect
	Update types.ObjectUpdate
}

func (pc *change) MarshalJSON() ([]byte, error) {
	if len(pc.cmd.kind) == 0 {
		return json.Marshal(pc.Update.ChangeSet)
	}

	return json.Marshal(pc.Update)
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

	if pc.cmd.obj != "" {
		fmt.Fprintf(pc.cmd.Out, "%s\t", pc.cmd.obj)
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

	for _, c := range pc.Update.ChangeSet {
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

		if len(pc.cmd.kind) != 0 {
			pc.cmd.obj = pc.Update.Obj.String()
		}

		if pc.cmd.single && rtype.Kind() == reflect.Struct && !rtype.Implements(stringer) {
			pc.writeStruct(c.Name, rval, rtype)
			continue
		}

		pc.output(c.Name, rval, rtype)
	}

	return tw.Flush()
}

func (pc *change) Dump() interface{} {
	if pc.cmd.simple && len(pc.Update.ChangeSet) == 1 {
		val := pc.Update.ChangeSet[0].Val
		if val != nil {
			rval := reflect.ValueOf(val)
			rtype := rval.Type()

			if strings.HasPrefix(rtype.Name(), "ArrayOf") {
				return rval.Field(0).Interface()
			}
		}

		return val
	}

	return pc.Update
}

func (cmd *collect) match(update types.ObjectUpdate) bool {
	if len(cmd.filter) == 0 {
		return false
	}

	for _, c := range update.ChangeSet {
		if cmd.filter.MatchProperty(types.DynamicProperty{Name: c.Name, Val: c.Val}) {
			return true
		}
	}

	return false
}

func (cmd *collect) toFilter(f *flag.FlagSet, props []string) ([]string, error) {
	// TODO: Only supporting 1 filter prop for now.  More than one would require some
	// accounting / accumulating of multiple updates.  And need to consider objects
	// then enter/leave a container view.
	if len(props) != 2 || !strings.HasPrefix(props[0], "-") {
		return props, nil
	}

	cmd.filter = property.Filter{props[0][1:]: props[1]}

	return cmd.filter.Keys(), nil
}

type dumpFilter struct {
	types.CreateFilter
}

func (f *dumpFilter) Dump() interface{} {
	return f.CreateFilter
}

// Write satisfies the flags.OutputWriter interface, but is not used with dumpFilter.
func (f *dumpFilter) Write(w io.Writer) error {
	return nil
}

func (cmd *collect) decodeFilter(filter *property.WaitFilter) error {
	var r io.Reader

	if cmd.raw == "-" {
		r = os.Stdin
	} else {
		f, err := os.Open(cmd.raw)
		if err != nil {
			return err
		}
		defer f.Close()
		r = f
	}

	env := soap.Envelope{
		Body: &methods.CreateFilterBody{Req: &filter.CreateFilter},
	}

	dec := xml.NewDecoder(r)
	dec.TypeFunc = types.TypeFunc()
	return dec.Decode(&env)
}

func (cmd *collect) Run(ctx context.Context, f *flag.FlagSet) error {
	client, err := cmd.Client()
	if err != nil {
		return err
	}

	p := property.DefaultCollector(client)
	filter := new(property.WaitFilter)

	if cmd.raw == "" {
		ref := vim25.ServiceInstance
		arg := f.Arg(0)

		if len(cmd.kind) != 0 {
			ref = client.ServiceContent.RootFolder
		}

		switch arg {
		case "", "-":
		default:
			ref, err = cmd.ManagedObject(ctx, arg)
			if err != nil {
				return err
			}
		}

		var props []string
		if f.NArg() > 1 {
			props = f.Args()[1:]
			cmd.single = len(props) == 1
		}

		props, err = cmd.toFilter(f, props)
		if err != nil {
			return err
		}

		if len(cmd.kind) == 0 {
			filter.Add(ref, ref.Type, props)
		} else {
			m := view.NewManager(client)

			v, cerr := m.CreateContainerView(ctx, ref, cmd.kind, true)
			if cerr != nil {
				return cerr
			}

			defer v.Destroy(ctx)

			for _, kind := range cmd.kind {
				filter.Add(v.Reference(), kind, props, v.TraversalSpec())
			}
		}
	} else {
		if err = cmd.decodeFilter(filter); err != nil {
			return err
		}
	}

	if cmd.dump {
		if !cmd.JSON {
			cmd.Dump = true
		}
		return cmd.WriteResult(&dumpFilter{filter.CreateFilter})
	}

	entered := false
	hasFilter := len(cmd.filter) != 0

	if cmd.wait != 0 {
		filter.Options = &types.WaitOptions{
			MaxWaitSeconds: types.NewInt32(int32(cmd.wait.Seconds())),
		}
	}

	return cmd.WithCancel(ctx, func(wctx context.Context) error {
		return property.WaitForUpdates(wctx, p, filter, func(updates []types.ObjectUpdate) bool {
			matches := 0
			for _, update := range updates {
				if entered && update.Kind == types.ObjectUpdateKindEnter {
					// on the first update we only get kind "enter"
					// if a new object is added, the next update with have both "enter" and "modify".
					continue
				}

				c := &change{cmd, update}

				if hasFilter {
					if cmd.match(update) {
						matches++
					} else {
						continue
					}
				}

				_ = cmd.WriteResult(c)
			}

			entered = true

			if hasFilter {
				if matches > 0 {
					return true
				}

				return false
			}

			cmd.n--

			return cmd.n == -1 && cmd.wait == 0
		})
	})
}
