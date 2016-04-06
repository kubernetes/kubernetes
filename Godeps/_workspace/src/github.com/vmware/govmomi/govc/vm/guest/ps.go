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
	"flag"
	"fmt"
	"os"
	"strconv"
	"text/tabwriter"

	"github.com/vmware/govmomi/govc/cli"
	"golang.org/x/net/context"
)

type ps struct {
	*GuestFlag

	every bool

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
	cmd.GuestFlag, ctx = newGuestFlag(ctx)
	cmd.GuestFlag.Register(ctx, f)

	cmd.uids = make(map[string]bool)
	f.BoolVar(&cmd.every, "e", false, "Select all processes")
	f.Var(&cmd.pids, "p", "Select by process ID")
	f.Var(&cmd.uids, "U", "Select by process UID")
}

func (cmd *ps) Process(ctx context.Context) error {
	if err := cmd.GuestFlag.Process(ctx); err != nil {
		return err
	}
	return nil
}

func (cmd *ps) Run(ctx context.Context, f *flag.FlagSet) error {
	m, err := cmd.ProcessManager()
	if err != nil {
		return err
	}

	if !cmd.every && len(cmd.uids) == 0 {
		cmd.uids[cmd.auth.Username] = true
	}

	procs, err := m.ListProcesses(context.TODO(), cmd.Auth(), cmd.pids)
	if err != nil {
		return err
	}

	tw := tabwriter.NewWriter(os.Stdout, 4, 0, 2, ' ', 0)

	fmt.Fprintf(tw, "%s\t%s\t%s\t%s\n", "UID", "PID", "STIME", "CMD")

	for _, p := range procs {
		if cmd.every || cmd.uids[p.Owner] {
			fmt.Fprintf(tw, "%s\t%d\t%s\t%s\n", p.Owner, p.Pid, p.StartTime.Format("15:04"), p.CmdLine)
		}
	}

	return tw.Flush()
}
