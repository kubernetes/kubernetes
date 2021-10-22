/*
Copyright (c) 2014-2016 VMware, Inc. All Rights Reserved.

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
	"io"
	"os"

	"github.com/vmware/govmomi/govc/cli"
	"github.com/vmware/govmomi/vim25/soap"
)

type upload struct {
	*GuestFlag
	*FileAttrFlag

	overwrite bool
}

func init() {
	cli.Register("guest.upload", &upload{})
}

func (cmd *upload) Register(ctx context.Context, f *flag.FlagSet) {
	cmd.GuestFlag, ctx = newGuestFlag(ctx)
	cmd.GuestFlag.Register(ctx, f)
	cmd.FileAttrFlag, ctx = newFileAttrFlag(ctx)
	cmd.FileAttrFlag.Register(ctx, f)

	f.BoolVar(&cmd.overwrite, "f", false, "If set, the guest destination file is clobbered")
}

func (cmd *upload) Usage() string {
	return "SOURCE DEST"
}

func (cmd *upload) Description() string {
	return `Copy SOURCE from the local system to DEST in the guest VM.

If SOURCE name is "-", read source from stdin.

Examples:
  govc guest.upload -l user:pass -vm=my-vm ~/.ssh/id_rsa.pub /home/$USER/.ssh/authorized_keys
  cowsay "have a great day" | govc guest.upload -l user:pass -vm=my-vm - /etc/motd`
}

func (cmd *upload) Process(ctx context.Context) error {
	if err := cmd.GuestFlag.Process(ctx); err != nil {
		return err
	}
	if err := cmd.FileAttrFlag.Process(ctx); err != nil {
		return err
	}
	return nil
}

func (cmd *upload) Run(ctx context.Context, f *flag.FlagSet) error {
	if f.NArg() != 2 {
		return flag.ErrHelp
	}

	c, err := cmd.Toolbox()
	if err != nil {
		return err
	}

	src := f.Arg(0)
	dst := f.Arg(1)

	p := soap.DefaultUpload

	var r io.Reader = os.Stdin

	if src != "-" {
		f, err := os.Open(src)
		if err != nil {
			return err
		}
		defer f.Close()

		r = f

		if cmd.OutputFlag.TTY {
			logger := cmd.ProgressLogger("Uploading... ")
			p.Progress = logger
			defer logger.Wait()
		}
	}

	return c.Upload(ctx, r, dst, p, cmd.Attr(), cmd.overwrite)
}
