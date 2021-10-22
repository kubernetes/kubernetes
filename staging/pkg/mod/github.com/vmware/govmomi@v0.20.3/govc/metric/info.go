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
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"strings"
	"text/tabwriter"

	"github.com/vmware/govmomi/govc/cli"
	"github.com/vmware/govmomi/vim25/types"
)

type info struct {
	*PerformanceFlag
}

func init() {
	cli.Register("metric.info", &info{})
}

func (cmd *info) Register(ctx context.Context, f *flag.FlagSet) {
	cmd.PerformanceFlag, ctx = NewPerformanceFlag(ctx)
	cmd.PerformanceFlag.Register(ctx, f)
}

func (cmd *info) Usage() string {
	return "PATH [NAME]..."
}

func (cmd *info) Description() string {
	return `Metric info for NAME.

If PATH is a value other than '-', provider summary and instance list are included
for the given object type.

If NAME is not specified, all available metrics for the given INTERVAL are listed.
An object PATH must be provided in this case.

Examples:
  govc metric.info vm/my-vm
  govc metric.info -i 300 vm/my-vm
  govc metric.info - cpu.usage.average
  govc metric.info /dc1/host/cluster cpu.usage.average`
}

func (cmd *info) Process(ctx context.Context) error {
	if err := cmd.PerformanceFlag.Process(ctx); err != nil {
		return err
	}
	return nil
}

type EntityDetail struct {
	Realtime   bool
	Historical bool
	Instance   []string
}

type MetricInfo struct {
	Counter          *types.PerfCounterInfo
	Enabled          []string
	PerDeviceEnabled []string
	Detail           *EntityDetail
}

type infoResult struct {
	Info []*MetricInfo
	cmd  *info
}

func (r *infoResult) Write(w io.Writer) error {
	tw := tabwriter.NewWriter(w, 2, 0, 2, ' ', 0)

	for _, info := range r.Info {
		counter := info.Counter

		fmt.Fprintf(tw, "Name:\t%s\n", counter.Name())
		fmt.Fprintf(tw, "  Label:\t%s\n", counter.NameInfo.GetElementDescription().Label)
		fmt.Fprintf(tw, "  Summary:\t%s\n", counter.NameInfo.GetElementDescription().Summary)
		fmt.Fprintf(tw, "  Group:\t%s\n", counter.GroupInfo.GetElementDescription().Label)
		fmt.Fprintf(tw, "  Unit:\t%s\n", counter.UnitInfo.GetElementDescription().Label)
		fmt.Fprintf(tw, "  Rollup type:\t%s\n", counter.RollupType)
		fmt.Fprintf(tw, "  Stats type:\t%s\n", counter.StatsType)
		fmt.Fprintf(tw, "  Level:\t%d\n", counter.Level)
		fmt.Fprintf(tw, "    Intervals:\t%s\n", strings.Join(info.Enabled, ","))
		fmt.Fprintf(tw, "  Per-device level:\t%d\n", counter.PerDeviceLevel)
		fmt.Fprintf(tw, "    Intervals:\t%s\n", strings.Join(info.PerDeviceEnabled, ","))

		summary := info.Detail
		if summary == nil {
			continue
		}

		fmt.Fprintf(tw, "  Realtime:\t%t\n", summary.Realtime)
		fmt.Fprintf(tw, "  Historical:\t%t\n", summary.Historical)
		fmt.Fprintf(tw, "  Instances:\t%s\n", strings.Join(summary.Instance, ","))
	}

	return tw.Flush()
}

func (r *infoResult) MarshalJSON() ([]byte, error) {
	m := make(map[string]*MetricInfo)

	for _, info := range r.Info {
		m[info.Counter.Name()] = info
	}

	return json.Marshal(m)
}

func (cmd *info) Run(ctx context.Context, f *flag.FlagSet) error {
	if f.NArg() == 0 {
		return flag.ErrHelp
	}

	names := f.Args()[1:]

	m, err := cmd.Manager(ctx)
	if err != nil {
		return err
	}

	counters, err := m.CounterInfoByName(ctx)
	if err != nil {
		return err
	}

	intervals, err := m.HistoricalInterval(ctx)
	if err != nil {
		return err
	}
	enabled := intervals.Enabled()

	var summary *types.PerfProviderSummary
	var mids map[int32][]*types.PerfMetricId

	if f.Arg(0) == "-" {
		if len(names) == 0 {
			return flag.ErrHelp
		}
	} else {
		objs, err := cmd.ManagedObjects(ctx, f.Args()[:1])
		if err != nil {
			return err
		}

		summary, err = m.ProviderSummary(ctx, objs[0])
		if err != nil {
			return err
		}

		all, err := m.AvailableMetric(ctx, objs[0], cmd.Interval(summary.RefreshRate))
		if err != nil {
			return err
		}

		mids = all.ByKey()

		if len(names) == 0 {
			nc, _ := m.CounterInfoByKey(ctx)

			for i := range all {
				id := &all[i]
				if id.Instance != "" {
					continue
				}

				names = append(names, nc[id.CounterId].Name())
			}
		}
	}

	var metrics []*MetricInfo

	for _, name := range names {
		counter, ok := counters[name]
		if !ok {
			return cmd.ErrNotFound(name)
		}

		info := &MetricInfo{
			Counter:          counter,
			Enabled:          enabled[counter.Level],
			PerDeviceEnabled: enabled[counter.PerDeviceLevel],
		}

		metrics = append(metrics, info)

		if summary == nil {
			continue
		}

		var instances []string

		for _, id := range mids[counter.Key] {
			if id.Instance != "" {
				instances = append(instances, id.Instance)
			}
		}

		info.Detail = &EntityDetail{
			Realtime:   summary.CurrentSupported,
			Historical: summary.SummarySupported,
			Instance:   instances,
		}

	}

	return cmd.WriteResult(&infoResult{metrics, cmd})
}
