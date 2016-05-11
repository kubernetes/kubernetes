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

package esxcli

import (
	"flag"
	"fmt"
	"os"
	"sort"
	"strings"
	"text/tabwriter"

	"golang.org/x/net/context"

	"github.com/vmware/govmomi/govc/cli"
	"github.com/vmware/govmomi/govc/flags"
)

type esxcli struct {
	*flags.HostSystemFlag

	hints bool
}

func init() {
	cli.Register("host.esxcli", &esxcli{})
}

func (cmd *esxcli) Usage() string {
	return "COMMAND [ARG]..."
}

func (cmd *esxcli) Register(ctx context.Context, f *flag.FlagSet) {
	cmd.HostSystemFlag, ctx = flags.NewHostSystemFlag(ctx)
	cmd.HostSystemFlag.Register(ctx, f)

	f.BoolVar(&cmd.hints, "hints", true, "Use command info hints when formatting output")
}

func (cmd *esxcli) Process(ctx context.Context) error {
	if err := cmd.HostSystemFlag.Process(ctx); err != nil {
		return err
	}
	return nil
}

func (cmd *esxcli) Run(ctx context.Context, f *flag.FlagSet) error {
	c, err := cmd.Client()
	if err != nil {
		return nil
	}

	host, err := cmd.HostSystem()
	if err != nil {
		return err
	}

	e, err := NewExecutor(c, host)
	if err != nil {
		return err
	}

	res, err := e.Run(f.Args())
	if err != nil {
		return err
	}

	if len(res.Values) == 0 {
		return nil
	}

	var formatType string
	if cmd.hints {
		formatType = res.Info.Hints.Formatter()
	}

	// TODO: OutputFlag / format options
	switch formatType {
	case "table":
		cmd.formatTable(res)
	default:
		cmd.formatSimple(res)
	}

	return nil
}

func (cmd *esxcli) formatSimple(res *Response) {
	var keys []string
	for key := range res.Values[0] {
		keys = append(keys, key)
	}
	sort.Strings(keys)

	tw := tabwriter.NewWriter(os.Stdout, 2, 0, 2, ' ', 0)

	for i, rv := range res.Values {
		if i > 0 {
			fmt.Fprintln(tw)
			_ = tw.Flush()
		}
		for _, key := range keys {
			fmt.Fprintf(tw, "%s:\t%s\n", key, strings.Join(rv[key], ", "))
		}
	}

	_ = tw.Flush()
}

func (cmd *esxcli) formatTable(res *Response) {
	fields := res.Info.Hints.Fields()

	tw := tabwriter.NewWriter(os.Stdout, len(fields), 0, 2, ' ', 0)

	var hr []string
	for _, name := range fields {
		hr = append(hr, strings.Repeat("-", len(name)))
	}

	fmt.Fprintln(tw, strings.Join(fields, "\t"))
	fmt.Fprintln(tw, strings.Join(hr, "\t"))

	for _, vals := range res.Values {
		var row []string

		for _, name := range fields {
			key := strings.Replace(name, " ", "", -1)
			if val, ok := vals[key]; ok {
				row = append(row, strings.Join(val, ", "))
			} else {
				row = append(row, "")
			}
		}

		fmt.Fprintln(tw, strings.Join(row, "\t"))
	}

	_ = tw.Flush()
}
