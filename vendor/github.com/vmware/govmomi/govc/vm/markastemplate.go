/*
Copyright (c) 2014-2015 VMware, Inc. All Rights Reserved.

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

package vm

import (
	"context"
	"flag"

	"github.com/vmware/govmomi/govc/cli"
	"github.com/vmware/govmomi/govc/flags"
)

type markastemplate struct {
	*flags.SearchFlag
}

func init() {
	cli.Register("vm.markastemplate", &markastemplate{})
}

func (cmd *markastemplate) Register(ctx context.Context, f *flag.FlagSet) {
	cmd.SearchFlag, ctx = flags.NewSearchFlag(ctx, flags.SearchVirtualMachines)
	cmd.SearchFlag.Register(ctx, f)
}

func (cmd *markastemplate) Process(ctx context.Context) error {
	if err := cmd.SearchFlag.Process(ctx); err != nil {
		return err
	}
	return nil
}

func (cmd *markastemplate) Usage() string {
	return "VM..."
}

func (cmd *markastemplate) Description() string {
	return `Mark VM as a virtual machine template.

Examples:
  govc vm.markastemplate $name`
}

func (cmd *markastemplate) Run(ctx context.Context, f *flag.FlagSet) error {
	vms, err := cmd.VirtualMachines(f.Args())
	if err != nil {
		return err
	}

	for _, vm := range vms {
		err := vm.MarkAsTemplate(ctx)
		if err != nil {
			return err
		}
	}

	return nil
}
