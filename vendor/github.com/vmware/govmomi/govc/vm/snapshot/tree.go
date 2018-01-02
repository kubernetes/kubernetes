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

package snapshot

import (
	"context"
	"flag"
	"fmt"
	"path"
	"strings"

	"github.com/vmware/govmomi/govc/cli"
	"github.com/vmware/govmomi/govc/flags"
	"github.com/vmware/govmomi/vim25/mo"
	"github.com/vmware/govmomi/vim25/types"
)

type tree struct {
	*flags.VirtualMachineFlag

	fullPath bool
	current  bool
	date     bool
	id       bool

	info *types.VirtualMachineSnapshotInfo
}

func init() {
	cli.Register("snapshot.tree", &tree{})
}

func (cmd *tree) Register(ctx context.Context, f *flag.FlagSet) {
	cmd.VirtualMachineFlag, ctx = flags.NewVirtualMachineFlag(ctx)
	cmd.VirtualMachineFlag.Register(ctx, f)

	f.BoolVar(&cmd.fullPath, "f", false, "Print the full path prefix for snapshot")
	f.BoolVar(&cmd.current, "c", true, "Print the current snapshot")
	f.BoolVar(&cmd.date, "D", false, "Print the snapshot creation date")
	f.BoolVar(&cmd.id, "i", false, "Print the snapshot id")
}

func (cmd *tree) Description() string {
	return `List VM snapshots in a tree-like format.

The command will exit 0 with no output if VM does not have any snapshots.

Examples:
  govc snapshot.tree -vm my-vm
  govc snapshot.tree -vm my-vm -D -i`
}

func (cmd *tree) Process(ctx context.Context) error {
	if err := cmd.VirtualMachineFlag.Process(ctx); err != nil {
		return err
	}
	return nil
}

func (cmd *tree) write(level int, parent string, st []types.VirtualMachineSnapshotTree) {
	for _, s := range st {
		sname := s.Name

		if cmd.fullPath && parent != "" {
			sname = path.Join(parent, sname)
		}

		names := []string{sname}

		if cmd.current && s.Snapshot == *cmd.info.CurrentSnapshot {
			names = append(names, ".")
		}

		for _, name := range names {
			var attr []string
			var meta string

			if cmd.id {
				attr = append(attr, s.Snapshot.Value)
			}

			if cmd.date {
				attr = append(attr, s.CreateTime.Format("Jan 2 15:04"))
			}

			if len(attr) > 0 {
				meta = fmt.Sprintf("[%s]  ", strings.Join(attr, " "))
			}

			fmt.Printf("%s%s%s\n", strings.Repeat(" ", level), meta, name)
		}

		cmd.write(level+2, sname, s.ChildSnapshotList)
	}
}

func (cmd *tree) Run(ctx context.Context, f *flag.FlagSet) error {
	if f.NArg() != 0 {
		return flag.ErrHelp
	}

	vm, err := cmd.VirtualMachine()
	if err != nil {
		return err
	}

	if vm == nil {
		return flag.ErrHelp
	}

	var o mo.VirtualMachine

	err = vm.Properties(ctx, vm.Reference(), []string{"snapshot"}, &o)
	if err != nil {
		return err
	}

	if o.Snapshot == nil {
		return nil
	}

	if cmd.current && o.Snapshot.CurrentSnapshot == nil {
		cmd.current = false
	}

	cmd.info = o.Snapshot

	cmd.write(0, "", o.Snapshot.RootSnapshotList)

	return nil
}
