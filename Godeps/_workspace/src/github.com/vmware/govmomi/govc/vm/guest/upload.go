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
	"os"

	"github.com/vmware/govmomi/govc/cli"
	"golang.org/x/net/context"
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
	m, err := cmd.FileManager()
	if err != nil {
		return err
	}

	src := f.Arg(0)
	dst := f.Arg(1)

	s, err := os.Stat(src)
	if err != nil {
		return err
	}

	url, err := m.InitiateFileTransferToGuest(context.TODO(), cmd.Auth(), dst, cmd.Attr(), s.Size(), cmd.overwrite)
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

	return c.Client.UploadFile(src, u, nil)
}
