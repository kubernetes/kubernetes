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
	"crypto/md5"
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path"
	"strings"
	"text/tabwriter"
	"time"

	"github.com/vmware/govmomi/govc/cli"
	"github.com/vmware/govmomi/performance"
	"github.com/vmware/govmomi/vim25/mo"
	"github.com/vmware/govmomi/vim25/types"
)

type sample struct {
	*PerformanceFlag

	d        int
	n        int
	t        bool
	plot     string
	instance string
}

func init() {
	cli.Register("metric.sample", &sample{})
}

func (cmd *sample) Register(ctx context.Context, f *flag.FlagSet) {
	cmd.PerformanceFlag, ctx = NewPerformanceFlag(ctx)
	cmd.PerformanceFlag.Register(ctx, f)

	f.IntVar(&cmd.d, "d", 30, "Limit object display name to D chars")
	f.IntVar(&cmd.n, "n", 6, "Max number of samples")
	f.StringVar(&cmd.plot, "plot", "", "Plot data using gnuplot")
	f.BoolVar(&cmd.t, "t", false, "Include sample times")
	f.StringVar(&cmd.instance, "instance", "*", "Instance")
}

func (cmd *sample) Usage() string {
	return "PATH... NAME..."
}

func (cmd *sample) Description() string {
	return `Sample for object PATH of metric NAME.

Interval ID defaults to 20 (realtime) if supported, otherwise 300 (5m interval).

By default, INSTANCE '*' samples all instances and the aggregate counter.
An INSTANCE value of '-' will only sample the aggregate counter.
An INSTANCE value other than '*' or '-' will only sample the given instance counter.

If PLOT value is set to '-', output a gnuplot script.  If non-empty with another
value, PLOT will pipe the script to gnuplot for you.  The value is also used to set
the gnuplot 'terminal' variable, unless the value is that of the DISPLAY env var.
Only 1 metric NAME can be specified when the PLOT flag is set.

Examples:
  govc metric.sample host/cluster1/* cpu.usage.average
  govc metric.sample -plot .png host/cluster1/* cpu.usage.average | xargs open
  govc metric.sample vm/* net.bytesTx.average net.bytesTx.average
  govc metric.sample -instance vmnic0 vm/* net.bytesTx.average
  govc metric.sample -instance - vm/* net.bytesTx.average`
}

func (cmd *sample) Process(ctx context.Context) error {
	if err := cmd.PerformanceFlag.Process(ctx); err != nil {
		return err
	}
	return nil
}

type sampleResult struct {
	cmd      *sample
	m        *performance.Manager
	counters map[string]*types.PerfCounterInfo
	Sample   []performance.EntityMetric
}

func (r *sampleResult) name(e types.ManagedObjectReference) string {
	var me mo.ManagedEntity
	_ = r.m.Properties(context.Background(), e, []string{"name"}, &me)

	name := me.Name

	if r.cmd.d > 0 && len(name) > r.cmd.d {
		return name[:r.cmd.d] + "*"
	}

	return name
}

func sampleInfoTimes(m *performance.EntityMetric) []string {
	vals := make([]string, len(m.SampleInfo))

	for i := range m.SampleInfo {
		vals[i] = m.SampleInfo[i].Timestamp.Format(time.RFC3339)
	}

	return vals
}

func (r *sampleResult) Plot(w io.Writer) error {
	if len(r.Sample) == 0 {
		return nil
	}

	if r.cmd.plot != "-" {
		cmd := exec.Command("gnuplot", "-persist")
		cmd.Stdout = w
		cmd.Stderr = os.Stderr
		stdin, err := cmd.StdinPipe()
		if err != nil {
			return err
		}

		if err = cmd.Start(); err != nil {
			return err
		}

		w = stdin
		defer func() {
			_ = stdin.Close()
			_ = cmd.Wait()
		}()
	}

	counter := r.counters[r.Sample[0].Value[0].Name]
	unit := counter.UnitInfo.GetElementDescription()

	fmt.Fprintf(w, "set title %q\n", counter.Name())
	fmt.Fprintf(w, "set ylabel %q\n", unit.Label)
	fmt.Fprintf(w, "set xlabel %q\n", "Time")
	fmt.Fprintf(w, "set xdata %s\n", "time")
	fmt.Fprintf(w, "set format x %q\n", "%H:%M")
	fmt.Fprintf(w, "set timefmt %q\n", "%Y-%m-%dT%H:%M:%SZ")

	ext := path.Ext(r.cmd.plot)
	if ext != "" {
		// If a file name is given, use the extension as terminal type.
		// If just an ext is given, use the entities and counter as the file name.
		file := r.cmd.plot
		name := r.cmd.plot[:len(r.cmd.plot)-len(ext)]
		r.cmd.plot = ext[1:]

		if name == "" {
			h := md5.New()

			for i := range r.Sample {
				_, _ = io.WriteString(h, r.Sample[i].Entity.String())
			}
			_, _ = io.WriteString(h, counter.Name())

			file = fmt.Sprintf("govc-plot-%x%s", h.Sum(nil), ext)
		}

		fmt.Fprintf(w, "set output %q\n", file)

		defer func() {
			fmt.Fprintln(r.cmd.Out, file)
		}()
	}

	switch r.cmd.plot {
	case "-", os.Getenv("DISPLAY"):
	default:
		fmt.Fprintf(w, "set terminal %s\n", r.cmd.plot)
	}

	if unit.Key == string(types.PerformanceManagerUnitPercent) {
		fmt.Fprintln(w, "set yrange [0:100]")
	}

	fmt.Fprintln(w)

	var set []string

	for i := range r.Sample {
		name := r.name(r.Sample[i].Entity)
		name = strings.Replace(name, "_", "*", -1) // underscore is some gnuplot markup?
		set = append(set, fmt.Sprintf("'-' using 1:2 title '%s' with lines", name))
	}

	fmt.Fprintf(w, "plot %s\n", strings.Join(set, ", "))

	for i := range r.Sample {
		times := sampleInfoTimes(&r.Sample[i])

		for _, value := range r.Sample[i].Value {
			for j := range value.Value {
				fmt.Fprintf(w, "%s %s\n", times[j], value.Format(value.Value[j]))
			}
		}

		fmt.Fprintln(w, "e")
	}

	return nil
}

func (r *sampleResult) Write(w io.Writer) error {
	if r.cmd.plot != "" {
		return r.Plot(w)
	}

	cmd := r.cmd
	tw := tabwriter.NewWriter(w, 2, 0, 2, ' ', 0)

	for i := range r.Sample {
		metric := r.Sample[i]
		name := r.name(metric.Entity)
		t := ""
		if cmd.t {
			t = metric.SampleInfoCSV()
		}

		for _, v := range metric.Value {
			counter := r.counters[v.Name]
			units := counter.UnitInfo.GetElementDescription().Label

			instance := v.Instance
			if instance == "" {
				instance = "-"
			}

			fmt.Fprintf(tw, "%s\t%s\t%s\t%v\t%s\t%s\n",
				name, instance, v.Name, t, v.ValueCSV(), units)
		}
	}

	return tw.Flush()
}

func (cmd *sample) Run(ctx context.Context, f *flag.FlagSet) error {
	m, err := cmd.Manager(ctx)
	if err != nil {
		return err
	}

	var paths []string
	var names []string

	byName, err := m.CounterInfoByName(ctx)
	if err != nil {
		return err
	}

	for _, arg := range f.Args() {
		if _, ok := byName[arg]; ok {
			names = append(names, arg)
		} else {
			paths = append(paths, arg)
		}
	}

	if len(paths) == 0 || len(names) == 0 {
		return flag.ErrHelp
	}

	if cmd.plot != "" {
		if len(names) > 1 {
			return flag.ErrHelp
		}

		if cmd.instance == "*" {
			cmd.instance = ""
		}
	}

	objs, err := cmd.ManagedObjects(ctx, paths)
	if err != nil {
		return err
	}

	s, err := m.ProviderSummary(ctx, objs[0])
	if err != nil {
		return err
	}

	if cmd.instance == "-" {
		cmd.instance = ""
	}

	spec := types.PerfQuerySpec{
		Format:     string(types.PerfFormatNormal),
		MaxSample:  int32(cmd.n),
		MetricId:   []types.PerfMetricId{{Instance: cmd.instance}},
		IntervalId: cmd.Interval(s.RefreshRate),
	}

	sample, err := m.SampleByName(ctx, spec, names, objs)
	if err != nil {
		return err
	}

	result, err := m.ToMetricSeries(ctx, sample)
	if err != nil {
		return err
	}

	counters, err := m.CounterInfoByName(ctx)
	if err != nil {
		return err
	}

	return cmd.WriteResult(&sampleResult{cmd, m, counters, result})
}
