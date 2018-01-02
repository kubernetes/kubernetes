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

	"github.com/vmware/govmomi/govc/cli"
	"github.com/vmware/govmomi/vim25/soap"
	"github.com/vmware/govmomi/vim25/types"
)

type mkdir struct {
	*GuestFlag

	createParents bool
}

func init() {
	cli.Register("guest.mkdir", &mkdir{})
}

func (cmd *mkdir) Register(ctx context.Context, f *flag.FlagSet) {
	cmd.GuestFlag, ctx = newGuestFlag(ctx)
	cmd.GuestFlag.Register(ctx, f)

	f.BoolVar(&cmd.createParents, "p", false, "Create intermediate directories as needed")
}

func (cmd *mkdir) Process(ctx context.Context) error {
	if err := cmd.GuestFlag.Process(ctx); err != nil {
		return err
	}
	return nil
}

func (cmd *mkdir) Run(ctx context.Context, f *flag.FlagSet) error {
	m, err := cmd.FileManager()
	if err != nil {
		return err
	}

	err = m.MakeDirectory(ctx, cmd.Auth(), f.Arg(0), cmd.createParents)

	// ignore EEXIST if -p flag is given
	if err != nil && cmd.createParents {
		if soap.IsSoapFault(err) {
			soapFault := soap.ToSoapFault(err)
			if _, ok := soapFault.VimFault().(types.FileAlreadyExists); ok {
				return nil
			}
		}
	}

	return err
}
