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

type rename struct {
	*flags.DatacenterFlag
}

func init() {
	cli.Register("object.rename", &rename{})
}

func (cmd *rename) Register(ctx context.Context, f *flag.FlagSet) {
	cmd.DatacenterFlag, ctx = flags.NewDatacenterFlag(ctx)
	cmd.DatacenterFlag.Register(ctx, f)
}

func (cmd *rename) Usage() string {
	return "PATH NAME"
}

func (cmd *rename) Description() string {
	return `Rename managed objects.

Examples:
  govc object.rename /dc1/network/dvs1 Switch1`
}

func (cmd *rename) Process(ctx context.Context) error {
	if err := cmd.DatacenterFlag.Process(ctx); err != nil {
		return err
	}
	return nil
}

func (cmd *rename) Run(ctx context.Context, f *flag.FlagSet) error {
	if f.NArg() != 2 {
		return flag.ErrHelp
	}

	c, err := cmd.Client()
	if err != nil {
		return err
	}

	objs, err := cmd.ManagedObjects(ctx, f.Args()[:1])
	if err != nil {
		return err
	}

	task, err := object.NewCommon(c, objs[0]).Rename(ctx, f.Arg(1))
	if err != nil {
		return err
	}

	logger := cmd.ProgressLogger(fmt.Sprintf("renaming %s... ", objs[0]))
	_, err = task.WaitForResult(ctx, logger)
	logger.Wait()

	return err
}
