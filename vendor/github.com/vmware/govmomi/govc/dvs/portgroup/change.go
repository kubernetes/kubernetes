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

package portgroup

import (
	"context"
	"flag"
	"fmt"

	"github.com/vmware/govmomi/govc/cli"
	"github.com/vmware/govmomi/govc/flags"
	"github.com/vmware/govmomi/object"
	"github.com/vmware/govmomi/vim25/mo"
)

type change struct {
	*flags.DatacenterFlag

	DVPortgroupConfigSpec
}

func init() {
	cli.Register("dvs.portgroup.change", &change{})
}

func (cmd *change) Register(ctx context.Context, f *flag.FlagSet) {
	cmd.DatacenterFlag, ctx = flags.NewDatacenterFlag(ctx)
	cmd.DatacenterFlag.Register(ctx, f)

	cmd.DVPortgroupConfigSpec.Register(ctx, f)
}

func (cmd *change) Description() string {
	return `Change DVS portgroup configuration.

Examples:
  govc dvs.portgroup.change -nports 26 ExternalNetwork
  govc dvs.portgroup.change -vlan 3214 ExternalNetwork`
}

func (cmd *change) Process(ctx context.Context) error {
	if err := cmd.DatacenterFlag.Process(ctx); err != nil {
		return err
	}
	return nil
}

func (cmd *change) Usage() string {
	return "PATH"
}

func (cmd *change) Run(ctx context.Context, f *flag.FlagSet) error {
	if f.NArg() != 1 {
		return flag.ErrHelp
	}

	path := f.Arg(0)

	finder, err := cmd.Finder()
	if err != nil {
		return err
	}

	net, err := finder.Network(ctx, path)
	if err != nil {
		return err
	}

	pg, ok := net.(*object.DistributedVirtualPortgroup)
	if !ok {
		return fmt.Errorf("%s (%T) is not of type %T", path, net, pg)
	}

	var s mo.DistributedVirtualPortgroup
	err = pg.Properties(ctx, pg.Reference(), []string{"config.configVersion"}, &s)
	if err != nil {
		return err
	}

	spec := cmd.Spec()
	spec.ConfigVersion = s.Config.ConfigVersion

	task, err := pg.Reconfigure(ctx, spec)
	if err != nil {
		return err
	}

	logger := cmd.ProgressLogger(fmt.Sprintf("changing %s portgroup configuration %s... ", pg.Name(), pg.InventoryPath))
	defer logger.Wait()

	_, err = task.WaitForResult(ctx, logger)
	return err
}
