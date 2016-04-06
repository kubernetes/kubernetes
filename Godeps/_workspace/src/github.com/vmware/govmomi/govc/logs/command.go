/*
Copyright (c) 2015-2016 VMware, Inc. All Rights Reserved.

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

package logs

import (
	"flag"
	"fmt"
	"math"

	"golang.org/x/net/context"

	"github.com/vmware/govmomi/govc/cli"
	"github.com/vmware/govmomi/govc/flags"
	"github.com/vmware/govmomi/object"
)

type logs struct {
	*flags.HostSystemFlag

	Max int32
	Key string
}

func init() {
	cli.Register("logs", &logs{})
}

func (cmd *logs) Register(ctx context.Context, f *flag.FlagSet) {
	cmd.HostSystemFlag, ctx = flags.NewHostSystemFlag(ctx)
	cmd.HostSystemFlag.Register(ctx, f)

	cmd.Max = 25 // default
	f.Var(flags.NewInt32(&cmd.Max), "n", "Output the last N logs")
	f.StringVar(&cmd.Key, "log", "", "Log file key")
}

func (cmd *logs) Process(ctx context.Context) error {
	if err := cmd.HostSystemFlag.Process(ctx); err != nil {
		return err
	}
	return nil
}

func (cmd *logs) Description() string {
	return `
The '-log' option defaults to "hostd" when connected directly to a host or
when connected to VirtualCenter and a '-host' option is given.  Otherwise,
the '-log' option defaults to "vpxd:vpxd.log".  The '-host' option is ignored
when connected directly to host.
See 'govc logs.ls' for other '-log' options.`
}

func (cmd *logs) Run(ctx context.Context, f *flag.FlagSet) error {
	c, err := cmd.Client()
	if err != nil {
		return err
	}

	defaultKey := "hostd"
	var host *object.HostSystem

	if c.IsVC() {
		host, err = cmd.HostSystemIfSpecified()
		if err != nil {
			return err
		}

		if host == nil {
			defaultKey = "vpxd:vpxd.log"
		}
	}

	m := object.NewDiagnosticManager(c)

	key := cmd.Key
	if key == "" {
		key = defaultKey
	}

	// get LineEnd without any LineText
	h, err := m.BrowseLog(ctx, host, key, math.MaxInt32, 0)
	if err != nil {
		return err
	}

	start := h.LineEnd - cmd.Max
	h, err = m.BrowseLog(ctx, host, key, start, 0)
	if err != nil {
		return err
	}

	for _, line := range h.LineText {
		fmt.Println(line)
	}

	return nil
}
