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

package object

import (
	"context"
	"flag"
	"fmt"

	"github.com/vmware/govmomi/govc/cli"
	"github.com/vmware/govmomi/govc/flags"
	"github.com/vmware/govmomi/object"
)

type destroy struct {
	*flags.DatacenterFlag
}

func init() {
	cli.Register("object.destroy", &destroy{})
}

func (cmd *destroy) Register(ctx context.Context, f *flag.FlagSet) {
	cmd.DatacenterFlag, ctx = flags.NewDatacenterFlag(ctx)
	cmd.DatacenterFlag.Register(ctx, f)
}

func (cmd *destroy) Usage() string {
	return "PATH..."
}

func (cmd *destroy) Description() string {
	return `Destroy managed objects.

Examples:
  govc object.destroy /dc1/network/dvs /dc1/host/cluster`
}

func (cmd *destroy) Process(ctx context.Context) error {
	if err := cmd.DatacenterFlag.Process(ctx); err != nil {
		return err
	}
	return nil
}

func (cmd *destroy) Run(ctx context.Context, f *flag.FlagSet) error {
	if f.NArg() == 0 {
		return flag.ErrHelp
	}

	c, err := cmd.Client()
	if err != nil {
		return err
	}

	objs, err := cmd.ManagedObjects(ctx, f.Args())
	if err != nil {
		return err
	}

	for _, obj := range objs {
		task, err := object.NewCommon(c, obj).Destroy(ctx)
		if err != nil {
			return err
		}

		logger := cmd.ProgressLogger(fmt.Sprintf("destroying %s... ", obj))
		_, err = task.WaitForResult(ctx, logger)
		logger.Wait()
		if err != nil {
			return err
		}
	}

	return nil
}
