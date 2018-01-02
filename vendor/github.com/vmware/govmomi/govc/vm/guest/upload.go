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
	"bytes"
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

	m, err := cmd.FileManager()
	if err != nil {
		return err
	}

	src := f.Arg(0)
	dst := f.Arg(1)

	var size int64
	var buf *bytes.Buffer

	if src == "-" {
		buf = new(bytes.Buffer)
		size, err = io.Copy(buf, os.Stdin)
		if err != nil {
			return err
		}
	} else {
		s, err := os.Stat(src)
		if err != nil {
			return err
		}
		size = s.Size()
	}

	url, err := m.InitiateFileTransferToGuest(ctx, cmd.Auth(), dst, cmd.Attr(), size, cmd.overwrite)
	if err != nil {
		return err
	}

	u, err := cmd.ParseURL(url)
	if err != nil {
		return err
	}

	c, err := cmd.Client()
	if err != nil {
		return nil
	}

	p := soap.DefaultUpload

	if buf != nil {
		p.ContentLength = size
		return c.Client.Upload(buf, u, &p)
	}

	if cmd.OutputFlag.TTY {
		logger := cmd.ProgressLogger("Uploading... ")
		p.Progress = logger
		defer logger.Wait()
	}

	return c.Client.UploadFile(src, u, nil)
}
