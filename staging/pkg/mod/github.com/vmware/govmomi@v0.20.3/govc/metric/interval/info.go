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

package interval

import (
	"context"
	"flag"
	"fmt"
	"text/tabwriter"
	"time"

	"github.com/vmware/govmomi/govc/cli"
	"github.com/vmware/govmomi/govc/metric"
)

type info struct {
	*metric.PerformanceFlag
}

func init() {
	cli.Register("metric.interval.info", &info{})
}

func (cmd *info) Register(ctx context.Context, f *flag.FlagSet) {
	cmd.PerformanceFlag, ctx = metric.NewPerformanceFlag(ctx)
	cmd.PerformanceFlag.Register(ctx, f)
}

func (cmd *info) Description() string {
	return `List historical metric intervals.

Examples:
  govc metric.interval.info
  govc metric.interval.info -i 300`
}

func (cmd *info) Process(ctx context.Context) error {
	if err := cmd.PerformanceFlag.Process(ctx); err != nil {
		return err
	}
	return nil
}

func (cmd *info) Run(ctx context.Context, f *flag.FlagSet) error {
	m, err := cmd.Manager(ctx)
	if err != nil {
		return err
	}

	intervals, err := m.HistoricalInterval(ctx)
	if err != nil {
		return err
	}

	tw := tabwriter.NewWriter(cmd.Out, 2, 0, 2, ' ', 0)
	cmd.Out = tw

	interval := cmd.Interval(0)

	for _, i := range intervals {
		if interval != 0 && i.SamplingPeriod != interval {
			continue
		}

		fmt.Fprintf(cmd.Out, "ID:\t%d\n", i.SamplingPeriod)
		fmt.Fprintf(cmd.Out, "  Enabled:\t%t\n", i.Enabled)
		fmt.Fprintf(cmd.Out, "  Interval:\t%s\n", time.Duration(i.SamplingPeriod)*time.Second)
		fmt.Fprintf(cmd.Out, "  Name:\t%s\n", i.Name)
		fmt.Fprintf(cmd.Out, "  Level:\t%d\n", i.Level)
	}

	return tw.Flush()
}
