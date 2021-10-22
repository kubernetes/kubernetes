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

package task

import (
	"context"
	"flag"
	"fmt"
	"strings"

	"github.com/vmware/govmomi/govc/cli"
	"github.com/vmware/govmomi/govc/flags"
	"github.com/vmware/govmomi/property"
	"github.com/vmware/govmomi/view"
	"github.com/vmware/govmomi/vim25/mo"
	"github.com/vmware/govmomi/vim25/types"
)

type recent struct {
	*flags.DatacenterFlag

	max    int
	follow bool
	long   bool
}

func init() {
	cli.Register("tasks", &recent{})
}

func (cmd *recent) Register(ctx context.Context, f *flag.FlagSet) {
	cmd.DatacenterFlag, ctx = flags.NewDatacenterFlag(ctx)
	cmd.DatacenterFlag.Register(ctx, f)

	f.IntVar(&cmd.max, "n", 25, "Output the last N tasks")
	f.BoolVar(&cmd.follow, "f", false, "Follow recent task updates")
	f.BoolVar(&cmd.long, "l", false, "Use long task description")
}

func (cmd *recent) Description() string {
	return `Display info for recent tasks.

When a task has completed, the result column includes the task duration on success or
error message on failure.  If a task is still in progress, the result column displays
the completion percentage and the task ID.  The task ID can be used as an argument to
the 'task.cancel' command.

By default, all recent tasks are included (via TaskManager), but can be limited by PATH
to a specific inventory object.

Examples:
  govc tasks
  govc tasks -f
  govc tasks -f /dc1/host/cluster1`
}

func (cmd *recent) Usage() string {
	return "[PATH]"
}

func (cmd *recent) Process(ctx context.Context) error {
	if err := cmd.DatacenterFlag.Process(ctx); err != nil {
		return err
	}
	return nil
}

func chop(s string) string {
	if len(s) < 30 {
		return s
	}

	return s[:29] + "*"
}

// taskName describes the tasks similar to the ESX ui
func taskName(info *types.TaskInfo) string {
	name := strings.TrimSuffix(info.Name, "_Task")
	switch name {
	case "":
		return info.DescriptionId
	case "Destroy", "Rename":
		return info.Entity.Type + "." + name
	default:
		return name
	}
}

func (cmd *recent) Run(ctx context.Context, f *flag.FlagSet) error {
	c, err := cmd.Client()
	if err != nil {
		return err
	}

	m := c.ServiceContent.TaskManager

	tn := taskName

	if cmd.long {
		var o mo.TaskManager
		err = property.DefaultCollector(c).RetrieveOne(ctx, *m, []string{"description.methodInfo"}, &o)
		if err != nil {
			return err
		}

		desc := make(map[string]string, len(o.Description.MethodInfo))

		for _, entry := range o.Description.MethodInfo {
			info := entry.GetElementDescription()
			desc[info.Key] = info.Label
		}

		tn = func(info *types.TaskInfo) string {
			if name, ok := desc[info.DescriptionId]; ok {
				return name
			}

			return taskName(info)
		}
	}

	watch := *m

	if f.NArg() == 1 {
		refs, merr := cmd.ManagedObjects(ctx, f.Args())
		if merr != nil {
			return merr
		}
		watch = refs[0]
	}

	v, err := view.NewManager(c).CreateTaskView(ctx, &watch)
	if err != nil {
		return nil
	}

	defer v.Destroy(context.Background())

	v.Follow = cmd.follow

	stamp := "15:04:05"
	tmpl := "%-40s %-30s %15s %9s %9s %9s %s\n"
	fmt.Fprintf(cmd.Out, tmpl, "Task", "Target", "Initiator", "Queued", "Started", "Completed", "Result")

	var last string
	updated := false

	return v.Collect(ctx, func(tasks []types.TaskInfo) {
		if !updated && len(tasks) > cmd.max {
			tasks = tasks[len(tasks)-cmd.max:]
		}
		updated = true

		for _, info := range tasks {
			var user string

			switch x := info.Reason.(type) {
			case *types.TaskReasonUser:
				user = x.UserName
			}

			if info.EntityName == "" || user == "" {
				continue
			}

			ruser := strings.SplitN(user, "\\", 2)
			if len(ruser) == 2 {
				user = ruser[1] // discard domain
			} else {
				user = strings.TrimPrefix(user, "com.vmware.") // e.g. com.vmware.vsan.health
			}

			queued := info.QueueTime.Format(stamp)
			start := "-"
			end := start

			if info.StartTime != nil {
				start = info.StartTime.Format(stamp)
			}

			msg := fmt.Sprintf("%2d%% %s", info.Progress, info.Task)

			if info.CompleteTime != nil {
				msg = info.CompleteTime.Sub(*info.StartTime).String()

				if info.State == types.TaskInfoStateError {
					msg = strings.TrimSuffix(info.Error.LocalizedMessage, ".")
				}

				end = info.CompleteTime.Format(stamp)
			}

			result := fmt.Sprintf("%-7s [%s]", info.State, msg)

			item := fmt.Sprintf(tmpl, tn(&info), chop(info.EntityName), user, queued, start, end, result)

			if item == last {
				continue // task info was updated, but the fields we display were not
			}
			last = item

			fmt.Fprint(cmd.Out, item)
		}
	})
}
