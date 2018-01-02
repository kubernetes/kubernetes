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

	"github.com/vmware/govmomi/govc/cli"
)

type kill struct {
	*GuestFlag

	pids pidSelector
}

func init() {
	cli.Register("guest.kill", &kill{})
}

func (cmd *kill) Register(ctx context.Context, f *flag.FlagSet) {
	cmd.GuestFlag, ctx = newGuestFlag(ctx)
	cmd.GuestFlag.Register(ctx, f)

	f.Var(&cmd.pids, "p", "Process ID")
}

func (cmd *kill) Process(ctx context.Context) error {
	if err := cmd.GuestFlag.Process(ctx); err != nil {
		return err
	}
	return nil
}

func (cmd *kill) Run(ctx context.Context, f *flag.FlagSet) error {
	m, err := cmd.ProcessManager()
	if err != nil {
		return err
	}

	for _, pid := range cmd.pids {
		if err := m.TerminateProcess(ctx, cmd.Auth(), pid); err != nil {
			return err
		}
	}

	return nil
}
