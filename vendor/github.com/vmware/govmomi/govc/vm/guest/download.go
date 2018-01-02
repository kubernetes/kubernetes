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
	"io"

	"context"
	"os"

	"github.com/vmware/govmomi/govc/cli"
	"github.com/vmware/govmomi/vim25/soap"
)

type download struct {
	*GuestFlag

	overwrite bool
}

func init() {
	cli.Register("guest.download", &download{})
}

func (cmd *download) Register(ctx context.Context, f *flag.FlagSet) {
	cmd.GuestFlag, ctx = newGuestFlag(ctx)
	cmd.GuestFlag.Register(ctx, f)

	f.BoolVar(&cmd.overwrite, "f", false, "If set, the local destination file is clobbered")
}

func (cmd *download) Usage() string {
	return "SOURCE DEST"
}

func (cmd *download) Description() string {
	return `Copy SOURCE from the guest VM to DEST on the local system.

If DEST name is "-", source is written to stdout.

Examples:
  govc guest.download -l user:pass -vm=my-vm /var/log/my.log ./local.log
  govc guest.download -l user:pass -vm=my-vm /etc/motd -`
}

func (cmd *download) Process(ctx context.Context) error {
	if err := cmd.GuestFlag.Process(ctx); err != nil {
		return err
	}
	return nil
}

func (cmd *download) Run(ctx context.Context, f *flag.FlagSet) error {
	if f.NArg() != 2 {
		return flag.ErrHelp
	}

	m, err := cmd.FileManager()
	if err != nil {
		return err
	}

	src := f.Arg(0)
	dst := f.Arg(1)

	_, err = os.Stat(dst)
	if err == nil && !cmd.overwrite {
		return os.ErrExist
	}

	info, err := m.InitiateFileTransferFromGuest(ctx, cmd.Auth(), src)
	if err != nil {
		return err
	}

	u, err := cmd.ParseURL(info.Url)
	if err != nil {
		return err
	}

	c, err := cmd.Client()
	if err != nil {
		return nil
	}

	p := soap.DefaultDownload

	if dst == "-" {
		f, _, err := c.Client.Download(u, &p)
		if err != nil {
			return err
		}
		_, err = io.Copy(os.Stdout, f)
		return err
	}

	if cmd.OutputFlag.TTY {
		logger := cmd.ProgressLogger("Downloading... ")
		p.Progress = logger
		defer logger.Wait()
	}

	return c.Client.DownloadFile(dst, u, &p)
}
