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

	"github.com/vmware/govmomi/govc/cli"
	"github.com/vmware/govmomi/govc/flags"
	"github.com/vmware/govmomi/object"
	"github.com/vmware/govmomi/vim25/types"
)

type cancel struct {
	*flags.ClientFlag
}

func init() {
	cli.Register("task.cancel", &cancel{})
}

func (cmd *cancel) Register(ctx context.Context, f *flag.FlagSet) {
	cmd.ClientFlag, ctx = flags.NewClientFlag(ctx)
	cmd.ClientFlag.Register(ctx, f)
}

func (cmd *cancel) Description() string {
	return `Cancel tasks.

Examples:
  govc task.cancel task-759`
}

func (cmd *cancel) Usage() string {
	return "ID..."
}

func (cmd *cancel) Process(ctx context.Context) error {
	if err := cmd.ClientFlag.Process(ctx); err != nil {
		return err
	}
	return nil
}

func (cmd *cancel) Run(ctx context.Context, f *flag.FlagSet) error {
	c, err := cmd.Client()
	if err != nil {
		return err
	}

	for _, id := range f.Args() {
		var ref types.ManagedObjectReference

		if !ref.FromString(id) {
			ref.Type = "Task"
			ref.Value = id
		}

		err = object.NewTask(c, ref).Cancel(ctx)
		if err != nil {
			return err
		}
	}

	return nil
}
