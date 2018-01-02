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

package rdm

import (
	"context"
	"flag"
	"fmt"
	"io"
	"os"
	"strings"
	"text/tabwriter"

	"github.com/vmware/govmomi/govc/cli"
	"github.com/vmware/govmomi/govc/flags"
	"github.com/vmware/govmomi/vim25/types"
)

type ls struct {
	*flags.VirtualMachineFlag
	*flags.OutputFlag
}

func init() {
	cli.Register("vm.rdm.ls", &ls{})
}

func (cmd *ls) Register(ctx context.Context, f *flag.FlagSet) {

	cmd.VirtualMachineFlag, ctx = flags.NewVirtualMachineFlag(ctx)
	cmd.VirtualMachineFlag.Register(ctx, f)
	cmd.OutputFlag, ctx = flags.NewOutputFlag(ctx)
	cmd.OutputFlag.Register(ctx, f)
}

func (cmd *ls) Description() string {
	return `List available devices that could be attach to VM with RDM.

Examples:
  govc vm.rdm.ls -vm VM`
}

func (cmd *ls) Process(ctx context.Context) error {

	if err := cmd.VirtualMachineFlag.Process(ctx); err != nil {
		return err
	}
	if err := cmd.OutputFlag.Process(ctx); err != nil {
		return err
	}
	return nil
}

func (cmd *ls) Run(ctx context.Context, f *flag.FlagSet) error {
	vm, err := cmd.VirtualMachine()
	if err != nil {
		return err
	}

	if vm == nil {
		return flag.ErrHelp
	}

	vmConfigOptions, err := vm.QueryConfigTarget(ctx)
	if err != nil {
		return err
	}

	res := infoResult{
		Disks: vmConfigOptions.ScsiDisk,
	}
	return cmd.WriteResult(&res)
}

type infoResult struct {
	Disks []types.VirtualMachineScsiDiskDeviceInfo
}

func (r *infoResult) Write(w io.Writer) error {
	tw := tabwriter.NewWriter(os.Stdout, 2, 0, 2, ' ', 0)
	for _, disk := range r.Disks {
		fmt.Fprintf(tw, "Name:\t%s\n", disk.Name)
		fmt.Fprintf(tw, "  Device name:\t%s\n", disk.Disk.DeviceName)
		fmt.Fprintf(tw, "  Device path:\t%s\n", disk.Disk.DevicePath)
		fmt.Fprintf(tw, "  Canonical Name:\t%s\n", disk.Disk.CanonicalName)

		var uids []string
		for _, descriptor := range disk.Disk.Descriptor {
			uids = append(uids, descriptor.Id)
		}

		fmt.Fprintf(tw, "  UIDS:\t%s\n", strings.Join(uids, " ,"))
	}
	return tw.Flush()
}
