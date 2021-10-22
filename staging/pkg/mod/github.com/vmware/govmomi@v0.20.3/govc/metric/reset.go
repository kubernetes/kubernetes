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

package metric

import (
	"context"
	"flag"

	"github.com/vmware/govmomi/govc/cli"
	"github.com/vmware/govmomi/vim25/methods"
	"github.com/vmware/govmomi/vim25/types"
)

type reset struct {
	*PerformanceFlag
}

func init() {
	cli.Register("metric.reset", &reset{})
}

func (cmd *reset) Register(ctx context.Context, f *flag.FlagSet) {
	cmd.PerformanceFlag, ctx = NewPerformanceFlag(ctx)
	cmd.PerformanceFlag.Register(ctx, f)
}

func (cmd *reset) Usage() string {
	return "NAME..."
}

func (cmd *reset) Description() string {
	return `Reset counter NAME to the default level of data collection.

Examples:
  govc metric.reset net.bytesRx.average net.bytesTx.average`
}

func (cmd *reset) Process(ctx context.Context) error {
	if err := cmd.PerformanceFlag.Process(ctx); err != nil {
		return err
	}
	return nil
}

func (cmd *reset) Run(ctx context.Context, f *flag.FlagSet) error {
	if f.NArg() == 0 {
		return flag.ErrHelp
	}

	m, err := cmd.Manager(ctx)
	if err != nil {
		return err
	}

	counters, err := m.CounterInfoByName(ctx)
	if err != nil {
		return err
	}

	var ids []int32

	for _, name := range f.Args() {
		counter, ok := counters[name]
		if !ok {
			return cmd.ErrNotFound(name)
		}

		ids = append(ids, counter.Key)
	}

	_, err = methods.ResetCounterLevelMapping(ctx, m.Client(), &types.ResetCounterLevelMapping{
		This:     m.Reference(),
		Counters: ids,
	})

	return err
}
