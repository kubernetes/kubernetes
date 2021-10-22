/*
Copyright (c) 2014-2017 VMware, Inc. All Rights Reserved.

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

	"github.com/vmware/govmomi/govc/cli"
)

type getenv struct {
	*GuestFlag
}

func init() {
	cli.Register("guest.getenv", &getenv{})
}

func (cmd *getenv) Register(ctx context.Context, f *flag.FlagSet) {
	cmd.GuestFlag, ctx = newGuestFlag(ctx)
	cmd.GuestFlag.Register(ctx, f)
}

func (cmd *getenv) Process(ctx context.Context) error {
	if err := cmd.GuestFlag.Process(ctx); err != nil {
		return err
	}
	return nil
}

func (cmd *getenv) Usage() string {
	return "[NAME]..."
}

func (cmd *getenv) Description() string {
	return `Read NAME environment variables from VM.

Examples:
  govc guest.getenv -vm $name
  govc guest.getenv -vm $name HOME`
}

func (cmd *getenv) Run(ctx context.Context, f *flag.FlagSet) error {
	m, err := cmd.ProcessManager()
	if err != nil {
		return err
	}

	vars, err := m.ReadEnvironmentVariable(ctx, cmd.Auth(), f.Args())
	if err != nil {
		return err
	}

	for _, v := range vars {
		fmt.Printf("%s\n", v)
	}

	return nil
}
