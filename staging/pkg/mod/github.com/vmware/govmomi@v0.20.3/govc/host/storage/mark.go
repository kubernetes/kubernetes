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

package storage

import (
	"context"
	"flag"
	"fmt"

	"github.com/vmware/govmomi/govc/cli"
	"github.com/vmware/govmomi/govc/flags"
	"github.com/vmware/govmomi/object"
	"github.com/vmware/govmomi/vim25/mo"
	"github.com/vmware/govmomi/vim25/types"
)

type mark struct {
	*flags.HostSystemFlag

	ssd   *bool
	local *bool
}

func init() {
	cli.Register("host.storage.mark", &mark{})
}

func (cmd *mark) Register(ctx context.Context, f *flag.FlagSet) {
	cmd.HostSystemFlag, ctx = flags.NewHostSystemFlag(ctx)
	cmd.HostSystemFlag.Register(ctx, f)

	f.Var(flags.NewOptionalBool(&cmd.ssd), "ssd", "Mark as SSD")
	f.Var(flags.NewOptionalBool(&cmd.local), "local", "Mark as local")
}

func (cmd *mark) Process(ctx context.Context) error {
	if err := cmd.HostSystemFlag.Process(ctx); err != nil {
		return err
	}
	return nil
}

func (cmd *mark) Usage() string {
	return "DEVICE_PATH"
}

func (cmd *mark) Description() string {
	return `Mark device at DEVICE_PATH.`
}

func (cmd *mark) Mark(ctx context.Context, ss *object.HostStorageSystem, uuid string) error {
	var err error
	var task *object.Task

	if cmd.ssd != nil {
		if *cmd.ssd {
			task, err = ss.MarkAsSsd(ctx, uuid)
		} else {
			task, err = ss.MarkAsNonSsd(ctx, uuid)
		}

		if err != nil {
			return err
		}

		err = task.Wait(ctx)
		if err != nil {
			return err
		}
	}

	if cmd.local != nil {
		if *cmd.local {
			task, err = ss.MarkAsLocal(ctx, uuid)
		} else {
			task, err = ss.MarkAsNonLocal(ctx, uuid)
		}

		if err != nil {
			return err
		}

		err = task.Wait(ctx)
		if err != nil {
			return err
		}
	}

	return nil
}

func (cmd *mark) Run(ctx context.Context, f *flag.FlagSet) error {
	if f.NArg() != 1 {
		return fmt.Errorf("specify device path")
	}

	path := f.Args()[0]

	host, err := cmd.HostSystem()
	if err != nil {
		return err
	}

	ss, err := host.ConfigManager().StorageSystem(ctx)
	if err != nil {
		return err
	}

	var hss mo.HostStorageSystem
	err = ss.Properties(ctx, ss.Reference(), nil, &hss)
	if err != nil {
		return nil
	}

	for _, e := range hss.StorageDeviceInfo.ScsiLun {
		disk, ok := e.(*types.HostScsiDisk)
		if !ok {
			continue
		}

		if disk.DevicePath == path {
			return cmd.Mark(ctx, ss, disk.Uuid)
		}
	}

	return fmt.Errorf("%s not found", path)
}
