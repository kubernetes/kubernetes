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

package guest

import (
	"context"
	"flag"
	"strconv"
	"strings"

	"github.com/vmware/govmomi/govc/cli"
	"github.com/vmware/govmomi/vim25/types"
)

type chown struct {
	*GuestFlag
}

func init() {
	cli.Register("guest.chown", &chown{})
}

func (cmd *chown) Register(ctx context.Context, f *flag.FlagSet) {
	cmd.GuestFlag, ctx = newGuestFlag(ctx)
	cmd.GuestFlag.Register(ctx, f)
}

func (cmd *chown) Process(ctx context.Context) error {
	if err := cmd.GuestFlag.Process(ctx); err != nil {
		return err
	}
	return nil
}

func (cmd *chown) Usage() string {
	return "UID[:GID] FILE"
}

func (cmd *chown) Description() string {
	return `Change FILE UID and GID on VM.

Examples:
  govc guest.chown -vm $name UID[:GID] /var/log/foo.log`
}

func (cmd *chown) Run(ctx context.Context, f *flag.FlagSet) error {
	if f.NArg() != 2 {
		return flag.ErrHelp
	}

	m, err := cmd.FileManager()
	if err != nil {
		return err
	}

	var attr types.GuestPosixFileAttributes

	ids := strings.SplitN(f.Arg(0), ":", 2)
	if len(ids) == 0 {
		return flag.ErrHelp
	}

	id, err := strconv.Atoi(ids[0])
	if err != nil {
		return err
	}

	attr.OwnerId = new(int32)
	*attr.OwnerId = int32(id)

	if len(ids) == 2 {
		id, err = strconv.Atoi(ids[1])
		if err != nil {
			return err
		}

		attr.GroupId = new(int32)
		*attr.GroupId = int32(id)
	}

	return m.ChangeFileAttributes(ctx, cmd.Auth(), f.Arg(1), &attr)
}
