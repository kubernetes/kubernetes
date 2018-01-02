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

package guest

import (
	"context"
	"flag"
	"fmt"
	"io"
	"os"
	"strconv"
	"text/tabwriter"
	"time"

	"github.com/vmware/govmomi/govc/cli"
	"github.com/vmware/govmomi/govc/flags"
	"github.com/vmware/govmomi/vim25/types"
)

type ps struct {
	*flags.OutputFlag
	*GuestFlag

	every bool
	wait  bool

	pids pidSelector
	uids uidSelector
}

type pidSelector []int64

func (s *pidSelector) String() string {
	return fmt.Sprint(*s)
}

func (s *pidSelector) Set(value string) error {
	v, err := strconv.ParseInt(value, 0, 64)
	if err != nil {
		return err
	}
	*s = append(*s, v)
	return nil
}

type uidSelector map[string]bool

func (s uidSelector) String() string {
	return ""
}

func (s uidSelector) Set(value string) error {
	s[value] = true
	return nil
}

func init() {
	cli.Register("guest.ps", &ps{})
}

func (cmd *ps) Register(ctx context.Context, f *flag.FlagSet) {
	cmd.OutputFlag, ctx = flags.NewOutputFlag(ctx)
	cmd.OutputFlag.Register(ctx, f)

	cmd.GuestFlag, ctx = newGuestFlag(ctx)
	cmd.GuestFlag.Register(ctx, f)

	cmd.uids = make(map[string]bool)
	f.BoolVar(&cmd.every, "e", false, "Select all processes")
	f.BoolVar(&cmd.wait, "X", false, "Wait for process to exit")
	f.Var(&cmd.pids, "p", "Select by process ID")
	f.Var(&cmd.uids, "U", "Select by process UID")
}

func (cmd *ps) Process(ctx context.Context) error {
	if err := cmd.OutputFlag.Process(ctx); err != nil {
		return err
	}
	if err := cmd.GuestFlag.Process(ctx); err != nil {
		return err
	}
	return nil
}

func running(procs []types.GuestProcessInfo) bool {
	for _, p := range procs {
		if p.EndTime == nil {
			return true
		}
	}
	return false
}

func (cmd *ps) list(ctx context.Context) ([]types.GuestProcessInfo, error) {
	m, err := cmd.ProcessManager()
	if err != nil {
		return nil, err
	}

	auth := cmd.Auth()

	for {
		procs, err := m.ListProcesses(ctx, auth, cmd.pids)
		if err != nil {
			return nil, err
		}

		if cmd.wait && running(procs) {
			<-time.After(time.Millisecond * 250)
			continue
		}

		return procs, nil
	}
}

func (cmd *ps) Run(ctx context.Context, f *flag.FlagSet) error {
	procs, err := cmd.list(ctx)
	if err != nil {
		return err
	}

	r := &psResult{cmd, procs}

	return cmd.WriteResult(r)
}

type psResult struct {
	cmd         *ps
	ProcessInfo []types.GuestProcessInfo
}

func (r *psResult) Write(w io.Writer) error {
	tw := tabwriter.NewWriter(os.Stdout, 4, 0, 2, ' ', 0)

	fmt.Fprintf(tw, "%s\t%s\t%s\t%s\n", "UID", "PID", "STIME", "CMD")

	if !r.cmd.every && len(r.cmd.uids) == 0 {
		r.cmd.uids[r.cmd.auth.Username] = true
	}

	for _, p := range r.ProcessInfo {
		if r.cmd.every || r.cmd.uids[p.Owner] {
			fmt.Fprintf(tw, "%s\t%d\t%s\t%s\n", p.Owner, p.Pid, p.StartTime.Format("15:04"), p.CmdLine)
		}
	}

	return tw.Flush()
}
