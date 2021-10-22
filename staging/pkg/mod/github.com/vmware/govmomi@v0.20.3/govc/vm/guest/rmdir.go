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

	"github.com/vmware/govmomi/govc/cli"
)

type rmdir struct {
	*GuestFlag

	recursive bool
}

func init() {
	cli.Register("guest.rmdir", &rmdir{})
}

func (cmd *rmdir) Register(ctx context.Context, f *flag.FlagSet) {
	cmd.GuestFlag, ctx = newGuestFlag(ctx)
	cmd.GuestFlag.Register(ctx, f)

	f.BoolVar(&cmd.recursive, "r", false, "Recursive removal")
}

func (cmd *rmdir) Process(ctx context.Context) error {
	if err := cmd.GuestFlag.Process(ctx); err != nil {
		return err
	}
	return nil
}

func (cmd *rmdir) Usage() string {
	return "PATH"
}

func (cmd *rmdir) Description() string {
	return `Remove directory PATH in VM.

Examples:
  govc guest.rmdir -vm $name /tmp/empty-dir
  govc guest.rmdir -vm $name -r /tmp/non-empty-dir`
}

func (cmd *rmdir) Run(ctx context.Context, f *flag.FlagSet) error {
	m, err := cmd.FileManager()
	if err != nil {
		return err
	}

	return m.DeleteDirectory(ctx, cmd.Auth(), f.Arg(0), cmd.recursive)
}
