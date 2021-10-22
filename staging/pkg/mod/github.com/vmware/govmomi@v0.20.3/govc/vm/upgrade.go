package vm

import (
	"context"
	"flag"
	"fmt"

	"github.com/vmware/govmomi/govc/cli"
	"github.com/vmware/govmomi/govc/flags"
	"github.com/vmware/govmomi/task"
	"github.com/vmware/govmomi/vim25/types"
)

type upgrade struct {
	*flags.VirtualMachineFlag
	version int
}

func init() {
	cli.Register("vm.upgrade", &upgrade{})
}

func isAlreadyUpgraded(err error) bool {
	if fault, ok := err.(task.Error); ok {
		_, ok = fault.Fault().(*types.AlreadyUpgraded)
		return ok
	}

	return false
}

func (cmd *upgrade) Register(ctx context.Context, f *flag.FlagSet) {
	cmd.VirtualMachineFlag, ctx = flags.NewVirtualMachineFlag(ctx)
	cmd.VirtualMachineFlag.Register(ctx, f)

	f.IntVar(&cmd.version, "version", 0, "Target vm hardware version, by default -- latest available")
}

func (cmd *upgrade) Process(ctx context.Context) error {
	if err := cmd.VirtualMachineFlag.Process(ctx); err != nil {
		return err
	}
	return nil
}

func (cmd *upgrade) Description() string {
	return `Upgrade VMs to latest hardware version

Examples:
  govc vm.upgrade -vm $vm_name
  govc vm.upgrade -version=$version -vm $vm_name
  govc vm.upgrade -version=$version -vm.uuid $vm_uuid`
}

func (cmd *upgrade) Run(ctx context.Context, f *flag.FlagSet) error {
	vm, err := cmd.VirtualMachine()
	if err != nil {
		return err
	}

	var version = ""
	if cmd.version != 0 {
		version = fmt.Sprintf("vmx-%02d", cmd.version)
	}

	task, err := vm.UpgradeVM(ctx, version)
	if err != nil {
		return err
	}
	err = task.Wait(ctx)
	if err != nil {
		if isAlreadyUpgraded(err) {
			fmt.Println(err.Error())
		} else {
			return err
		}
	}

	return nil
}
