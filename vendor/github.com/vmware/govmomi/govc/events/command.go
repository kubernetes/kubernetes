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

package events

import (
	"context"
	"flag"
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/vmware/govmomi/event"
	"github.com/vmware/govmomi/govc/cli"
	"github.com/vmware/govmomi/govc/flags"
	"github.com/vmware/govmomi/vim25/types"
)

type events struct {
	*flags.DatacenterFlag

	Max   int32
	Tail  bool
	Force bool
}

func init() {
	// initialize with the maximum allowed objects set
	cli.Register("events", &events{})
}

func (cmd *events) Register(ctx context.Context, f *flag.FlagSet) {
	cmd.DatacenterFlag, ctx = flags.NewDatacenterFlag(ctx)
	cmd.DatacenterFlag.Register(ctx, f)

	cmd.Max = 25 // default
	f.Var(flags.NewInt32(&cmd.Max), "n", "Output the last N events")
	f.BoolVar(&cmd.Tail, "f", false, "Follow event stream")
	f.BoolVar(&cmd.Force, "force", false, "Disable number objects to monitor limit")
}

func (cmd *events) Description() string {
	return `Display events.

Examples:
  govc events vm/my-vm1 vm/my-vm2
  govc events /dc1/vm/* /dc2/vm/*
  govc ls -t HostSystem host/* | xargs govc events | grep -i vsan`
}

func (cmd *events) Usage() string {
	return "[PATH]..."
}

func (cmd *events) Process(ctx context.Context) error {
	if err := cmd.DatacenterFlag.Process(ctx); err != nil {
		return err
	}
	return nil
}

func (cmd *events) printEvents(ctx context.Context, obj *types.ManagedObjectReference, page []types.BaseEvent, m *event.Manager) error {
	event.Sort(page)
	if obj != nil {
		// print the object reference
		fmt.Fprintf(os.Stdout, "\n==> %s <==\n", obj.String())
	}
	for _, e := range page {
		cat, err := m.EventCategory(ctx, e)
		if err != nil {
			return err
		}

		event := e.GetEvent()
		msg := strings.TrimSpace(event.FullFormattedMessage)

		// if this is a TaskEvent gather a little more information
		if t, ok := e.(*types.TaskEvent); ok {
			// some tasks won't have this information, so just use the event message
			if t.Info.Entity != nil {
				msg = fmt.Sprintf("%s (target=%s %s)", msg, t.Info.Entity.Type, t.Info.EntityName)
			}
		}

		fmt.Fprintf(os.Stdout, "[%s] [%s] %s\n",
			event.CreatedTime.Local().Format(time.ANSIC),
			cat,
			msg)
	}
	return nil
}

func (cmd *events) Run(ctx context.Context, f *flag.FlagSet) error {
	c, err := cmd.Client()
	if err != nil {
		return err
	}

	objs, err := cmd.ManagedObjects(ctx, f.Args())
	if err != nil {
		return err
	}

	if len(objs) > 0 {
		// need an event manager
		m := event.NewManager(c)

		// get the event stream
		err = m.Events(ctx, objs, cmd.Max, cmd.Tail, cmd.Force, func(obj types.ManagedObjectReference, ee []types.BaseEvent) error {
			var o *types.ManagedObjectReference
			if len(objs) > 1 {
				o = &obj
			}
			err = cmd.printEvents(ctx, o, ee, m)
			if err != nil {
				return err
			}
			return nil
		})

		if err != nil {
			return err
		}

	}

	return nil
}
