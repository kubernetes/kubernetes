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

package storage

import (
	"context"
	"flag"
	"fmt"
	"io"
	"strings"
	"text/tabwriter"

	"github.com/vmware/govmomi/govc/cli"
	"github.com/vmware/govmomi/govc/flags"
	"github.com/vmware/govmomi/units"
	"github.com/vmware/govmomi/vim25/mo"
	"github.com/vmware/govmomi/vim25/types"
)

var infoTypes = []string{"hba", "lun"}

type infoType string

func (t *infoType) Set(s string) error {
	s = strings.ToLower(s)

	for _, e := range infoTypes {
		if s == e {
			*t = infoType(s)
			return nil
		}
	}

	return fmt.Errorf("invalid type")
}

func (t *infoType) String() string {
	return string(*t)
}

func (t *infoType) Result(hss mo.HostStorageSystem) flags.OutputWriter {
	switch string(*t) {
	case "hba":
		return hbaResult(hss)
	case "lun":
		return lunResult(hss)
	default:
		panic("unsupported")
	}
}

type info struct {
	*flags.HostSystemFlag
	*flags.OutputFlag

	typ        infoType
	rescan     bool
	refresh    bool
	rescanvmfs bool
	unclaimed  bool
}

func init() {
	cli.Register("host.storage.info", &info{})
}

func (cmd *info) Register(ctx context.Context, f *flag.FlagSet) {
	cmd.HostSystemFlag, ctx = flags.NewHostSystemFlag(ctx)
	cmd.HostSystemFlag.Register(ctx, f)
	cmd.OutputFlag, ctx = flags.NewOutputFlag(ctx)
	cmd.OutputFlag.Register(ctx, f)

	err := cmd.typ.Set("lun")
	if err != nil {
		panic(err)
	}

	f.Var(&cmd.typ, "t", fmt.Sprintf("Type (%s)", strings.Join(infoTypes, ",")))

	f.BoolVar(&cmd.rescan, "rescan", false, "Rescan all host bus adapters")
	f.BoolVar(&cmd.refresh, "refresh", false, "Refresh the storage system provider")
	f.BoolVar(&cmd.rescanvmfs, "rescan-vmfs", false, "Rescan for new VMFSs")
	f.BoolVar(&cmd.unclaimed, "unclaimed", false, "Only show disks that can be used as new VMFS datastores")
}

func (cmd *info) Description() string {
	return `Show HOST storage system information.

Examples:
  govc find / -type h | xargs -n1 govc host.storage.info -unclaimed -host`
}

func (cmd *info) Process(ctx context.Context) error {
	if err := cmd.HostSystemFlag.Process(ctx); err != nil {
		return err
	}
	if err := cmd.OutputFlag.Process(ctx); err != nil {
		return err
	}
	return nil
}

func (cmd *info) Run(ctx context.Context, f *flag.FlagSet) error {
	host, err := cmd.HostSystem()
	if err != nil {
		return err
	}

	ss, err := host.ConfigManager().StorageSystem(ctx)
	if err != nil {
		return err
	}

	if cmd.rescan {
		err = ss.RescanAllHba(ctx)
		if err != nil {
			return err
		}
	}

	if cmd.refresh {
		err = ss.Refresh(ctx)
		if err != nil {
			return err
		}
	}

	if cmd.rescanvmfs {
		err = ss.RescanVmfs(ctx)
		if err != nil {
			return err
		}
	}

	var hss mo.HostStorageSystem
	err = ss.Properties(ctx, ss.Reference(), nil, &hss)
	if err != nil {
		return nil
	}

	if cmd.unclaimed {
		ds, err := host.ConfigManager().DatastoreSystem(ctx)
		if err != nil {
			return err
		}

		disks, err := ds.QueryAvailableDisksForVmfs(ctx)
		if err != nil {
			return err
		}

		var luns []types.BaseScsiLun
		for i := range disks {
			luns = append(luns, &disks[i])
		}
		hss.StorageDeviceInfo.ScsiLun = luns
	}

	return cmd.WriteResult(cmd.typ.Result(hss))
}

type hbaResult mo.HostStorageSystem

func (r hbaResult) Write(w io.Writer) error {
	tw := tabwriter.NewWriter(w, 2, 0, 2, ' ', 0)

	fmt.Fprintf(tw, "Device\t")
	fmt.Fprintf(tw, "PCI\t")
	fmt.Fprintf(tw, "Driver\t")
	fmt.Fprintf(tw, "Status\t")
	fmt.Fprintf(tw, "Model\t")
	fmt.Fprintf(tw, "\n")

	for _, e := range r.StorageDeviceInfo.HostBusAdapter {
		hba := e.GetHostHostBusAdapter()

		fmt.Fprintf(tw, "%s\t", hba.Device)
		fmt.Fprintf(tw, "%s\t", hba.Pci)
		fmt.Fprintf(tw, "%s\t", hba.Driver)
		fmt.Fprintf(tw, "%s\t", hba.Status)
		fmt.Fprintf(tw, "%s\t", hba.Model)
		fmt.Fprintf(tw, "\n")
	}

	return tw.Flush()
}

type lunResult mo.HostStorageSystem

func (r lunResult) Write(w io.Writer) error {
	tw := tabwriter.NewWriter(w, 2, 0, 2, ' ', 0)

	fmt.Fprintf(tw, "Name\t")
	fmt.Fprintf(tw, "Type\t")
	fmt.Fprintf(tw, "Capacity\t")
	fmt.Fprintf(tw, "Model\t")
	fmt.Fprintf(tw, "\n")

	for _, e := range r.StorageDeviceInfo.ScsiLun {
		var tags []string
		var capacity int64

		lun := e.GetScsiLun()
		if disk, ok := e.(*types.HostScsiDisk); ok {
			capacity = int64(disk.Capacity.Block) * int64(disk.Capacity.BlockSize)
			if disk.LocalDisk != nil && *disk.LocalDisk {
				tags = append(tags, "local")
			}
			if disk.Ssd != nil && *disk.Ssd {
				tags = append(tags, "ssd")
			}
		}

		fmt.Fprintf(tw, "%s\t", lun.DeviceName)
		fmt.Fprintf(tw, "%s\t", lun.DeviceType)

		if capacity == 0 {
			fmt.Fprintf(tw, "-\t")
		} else {
			fmt.Fprintf(tw, "%s\t", units.ByteSize(capacity))
		}

		fmt.Fprintf(tw, "%s", lun.Model)
		if len(tags) > 0 {
			fmt.Fprintf(tw, " (%s)", strings.Join(tags, ","))
		}
		fmt.Fprintf(tw, "\n")
	}

	return tw.Flush()
}
