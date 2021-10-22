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
	"bytes"
	"context"
	"flag"
	"time"

	"github.com/vmware/govmomi/govc/cli"
	"github.com/vmware/govmomi/vim25/soap"
	"github.com/vmware/govmomi/vim25/types"
)

type touch struct {
	*GuestFlag

	nocreate bool
	atime    bool
	date     string
}

func init() {
	cli.Register("guest.touch", &touch{})
}

func (cmd *touch) Register(ctx context.Context, f *flag.FlagSet) {
	cmd.GuestFlag, ctx = newGuestFlag(ctx)
	cmd.GuestFlag.Register(ctx, f)

	f.BoolVar(&cmd.atime, "a", false, "Change only the access time")
	f.BoolVar(&cmd.nocreate, "c", false, "Do not create any files")
	f.StringVar(&cmd.date, "d", "", "Use DATE instead of current time")
}

func (cmd *touch) Process(ctx context.Context) error {
	if err := cmd.GuestFlag.Process(ctx); err != nil {
		return err
	}
	return nil
}

func (cmd *touch) Usage() string {
	return "FILE"
}

func (cmd *touch) Description() string {
	return `Change FILE times on VM.

Examples:
  govc guest.touch -vm $name /var/log/foo.log
  govc guest.touch -vm $name -d "$(date -d '1 day ago')" /var/log/foo.log`
}

func (cmd *touch) Run(ctx context.Context, f *flag.FlagSet) error {
	if f.NArg() != 1 {
		return flag.ErrHelp
	}

	m, err := cmd.FileManager()
	if err != nil {
		return err
	}

	name := f.Arg(0)

	var attr types.GuestFileAttributes
	now := time.Now()

	if cmd.date != "" {
		now, err = time.Parse(time.UnixDate, cmd.date)
		if err != nil {
			return err
		}
	}

	if cmd.atime {
		attr.AccessTime = &now
	} else {
		attr.ModificationTime = &now
	}

	err = m.ChangeFileAttributes(ctx, cmd.Auth(), name, &attr)
	if err != nil && !cmd.nocreate && soap.IsSoapFault(err) {
		fault := soap.ToSoapFault(err)
		if _, ok := fault.VimFault().(types.FileNotFound); ok {
			// create a new empty file
			url, cerr := m.InitiateFileTransferToGuest(ctx, cmd.Auth(), name, &attr, 0, false)
			if cerr != nil {
				return cerr
			}

			u, cerr := cmd.ParseURL(url)
			if cerr != nil {
				return cerr
			}

			c, cerr := cmd.Client()
			if cerr != nil {
				return cerr
			}

			err = c.Client.Upload(ctx, new(bytes.Buffer), u, &soap.DefaultUpload)
			if err == nil && cmd.date != "" {
				err = m.ChangeFileAttributes(ctx, cmd.Auth(), name, &attr)
			}
		}
	}

	return err
}
