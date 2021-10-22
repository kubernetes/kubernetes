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

	"github.com/vmware/govmomi/govc/cli"
	"github.com/vmware/govmomi/vim25/soap"
	"github.com/vmware/govmomi/vim25/types"
)

type mv struct {
	*GuestFlag

	noclobber bool
}

func init() {
	cli.Register("guest.mv", &mv{})
}

func (cmd *mv) Register(ctx context.Context, f *flag.FlagSet) {
	cmd.GuestFlag, ctx = newGuestFlag(ctx)
	cmd.GuestFlag.Register(ctx, f)

	f.BoolVar(&cmd.noclobber, "n", false, "Do not overwrite an existing file")
}

func (cmd *mv) Process(ctx context.Context) error {
	if err := cmd.GuestFlag.Process(ctx); err != nil {
		return err
	}
	return nil
}

func (cmd *mv) Usage() string {
	return "SOURCE DEST"
}

func (cmd *mv) Description() string {
	return `Move (rename) files in VM.

Examples:
  govc guest.mv -vm $name /tmp/foo.sh /tmp/bar.sh
  govc guest.mv -vm $name -n /tmp/baz.sh /tmp/bar.sh`
}

func (cmd *mv) Run(ctx context.Context, f *flag.FlagSet) error {
	m, err := cmd.FileManager()
	if err != nil {
		return err
	}

	src := f.Arg(0)
	dst := f.Arg(1)

	err = m.MoveFile(ctx, cmd.Auth(), src, dst, !cmd.noclobber)

	if err != nil {
		if soap.IsSoapFault(err) {
			soapFault := soap.ToSoapFault(err)
			if _, ok := soapFault.VimFault().(types.NotAFile); ok {
				err = m.MoveDirectory(ctx, cmd.Auth(), src, dst)
			}
		}
	}

	return err
}
