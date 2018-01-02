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
	"strings"

	"github.com/vmware/govmomi/govc/cli"
	"github.com/vmware/govmomi/vim25/types"
)

type start struct {
	*GuestFlag

	dir  string
	vars env
}

type env []string

func (e *env) String() string {
	return fmt.Sprint(*e)
}

func (e *env) Set(value string) error {
	*e = append(*e, value)
	return nil
}

func init() {
	cli.Register("guest.start", &start{})
}

func (cmd *start) Register(ctx context.Context, f *flag.FlagSet) {
	cmd.GuestFlag, ctx = newGuestFlag(ctx)
	cmd.GuestFlag.Register(ctx, f)

	f.StringVar(&cmd.dir, "C", "", "The absolute path of the working directory for the program to start")
	f.Var(&cmd.vars, "e", "Set environment variable (key=val)")
}

func (cmd *start) Process(ctx context.Context) error {
	if err := cmd.GuestFlag.Process(ctx); err != nil {
		return err
	}
	return nil
}

func (cmd *start) Run(ctx context.Context, f *flag.FlagSet) error {
	m, err := cmd.ProcessManager()
	if err != nil {
		return err
	}

	spec := types.GuestProgramSpec{
		ProgramPath:      f.Arg(0),
		Arguments:        strings.Join(f.Args()[1:], " "),
		WorkingDirectory: cmd.dir,
		EnvVariables:     cmd.vars,
	}

	pid, err := m.StartProgram(ctx, cmd.Auth(), &spec)
	if err != nil {
		return err
	}

	fmt.Printf("%d\n", pid)

	return nil
}
