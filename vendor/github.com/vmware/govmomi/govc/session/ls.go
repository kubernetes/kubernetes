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

package session

import (
	"context"
	"flag"
	"fmt"
	"io"
	"text/tabwriter"
	"time"

	"github.com/vmware/govmomi/govc/cli"
	"github.com/vmware/govmomi/govc/flags"
	"github.com/vmware/govmomi/property"
	"github.com/vmware/govmomi/vim25/mo"
)

type ls struct {
	*flags.ClientFlag
	*flags.OutputFlag
}

func init() {
	cli.Register("session.ls", &ls{})
}

func (cmd *ls) Register(ctx context.Context, f *flag.FlagSet) {
	cmd.ClientFlag, ctx = flags.NewClientFlag(ctx)
	cmd.ClientFlag.Register(ctx, f)

	cmd.OutputFlag, ctx = flags.NewOutputFlag(ctx)
	cmd.OutputFlag.Register(ctx, f)
}

func (cmd *ls) Description() string {
	return `List active sessions.

Examples:
  govc session.ls
  govc session.ls -json | jq -r .CurrentSession.Key`
}

func (cmd *ls) Process(ctx context.Context) error {
	if err := cmd.ClientFlag.Process(ctx); err != nil {
		return err
	}
	if err := cmd.OutputFlag.Process(ctx); err != nil {
		return err
	}
	return nil
}

type sessionInfo struct {
	cmd *ls
	mo.SessionManager
}

func (s *sessionInfo) Write(w io.Writer) error {
	tw := tabwriter.NewWriter(w, 4, 0, 2, ' ', 0)

	fmt.Fprintf(tw, "Key\t")
	fmt.Fprintf(tw, "Name\t")
	fmt.Fprintf(tw, "Time\t")
	fmt.Fprintf(tw, "Idle\t")
	fmt.Fprintf(tw, "Host\t")
	fmt.Fprintf(tw, "Agent\t")
	fmt.Fprintf(tw, "\t")
	fmt.Fprint(tw, "\n")

	for _, v := range s.SessionList {
		idle := "  ."
		if v.Key != s.CurrentSession.Key {
			since := time.Since(v.LastActiveTime)
			if since > time.Hour {
				idle = "old"
			} else {
				idle = (time.Duration(since.Seconds()) * time.Second).String()
			}
		}
		fmt.Fprintf(tw, "%s\t", v.Key)
		fmt.Fprintf(tw, "%s\t", v.UserName)
		fmt.Fprintf(tw, "%s\t", v.LoginTime.Format("2006-01-02 15:04"))
		fmt.Fprintf(tw, "%s\t", idle)
		fmt.Fprintf(tw, "%s\t", v.IpAddress)
		fmt.Fprintf(tw, "%s\t", v.UserAgent)
		fmt.Fprint(tw, "\n")
	}

	return tw.Flush()
}

func (cmd *ls) Run(ctx context.Context, f *flag.FlagSet) error {
	c, err := cmd.Client()
	if err != nil {
		return err
	}

	var m mo.SessionManager
	pc := property.DefaultCollector(c)
	err = pc.RetrieveOne(ctx, *c.ServiceContent.SessionManager, nil, &m)
	if err != nil {
		return nil
	}

	return cmd.WriteResult(&sessionInfo{cmd, m})
}
