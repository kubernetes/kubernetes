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

package datastore

import (
	"context"
	"errors"
	"flag"
	"fmt"

	"github.com/vmware/govmomi/govc/cli"
	"github.com/vmware/govmomi/govc/flags"
	"github.com/vmware/govmomi/object"
	"github.com/vmware/govmomi/vim25/soap"
	"github.com/vmware/govmomi/vim25/types"
)

type mkdir struct {
	*flags.DatastoreFlag

	createParents bool
	isNamespace   bool
}

func init() {
	cli.Register("datastore.mkdir", &mkdir{})
}

func (cmd *mkdir) Register(ctx context.Context, f *flag.FlagSet) {
	cmd.DatastoreFlag, ctx = flags.NewDatastoreFlag(ctx)
	cmd.DatastoreFlag.Register(ctx, f)

	f.BoolVar(&cmd.createParents, "p", false, "Create intermediate directories as needed")
	f.BoolVar(&cmd.isNamespace, "namespace", false, "Return uuid of namespace created on vsan datastore")
}

func (cmd *mkdir) Process(ctx context.Context) error {
	if err := cmd.DatastoreFlag.Process(ctx); err != nil {
		return err
	}
	return nil
}

func (cmd *mkdir) Usage() string {
	return "DIRECTORY"
}

func (cmd *mkdir) Run(ctx context.Context, f *flag.FlagSet) error {
	args := f.Args()
	if len(args) == 0 {
		return errors.New("missing operand")
	}

	c, err := cmd.Client()
	if err != nil {
		return err
	}

	if cmd.isNamespace {
		var uuid string
		var ds *object.Datastore

		if ds, err = cmd.Datastore(); err != nil {
			return err
		}

		path := args[0]

		nm := object.NewDatastoreNamespaceManager(c)
		if uuid, err = nm.CreateDirectory(ctx, ds, path, ""); err != nil {
			return err
		}

		fmt.Println(uuid)
	} else {
		var dc *object.Datacenter
		var path string

		dc, err = cmd.Datacenter()
		if err != nil {
			return err
		}

		path, err = cmd.DatastorePath(args[0])
		if err != nil {
			return err
		}

		m := object.NewFileManager(c)
		err = m.MakeDirectory(ctx, path, dc, cmd.createParents)

		// ignore EEXIST if -p flag is given
		if err != nil && cmd.createParents {
			if soap.IsSoapFault(err) {
				soapFault := soap.ToSoapFault(err)
				if _, ok := soapFault.VimFault().(types.FileAlreadyExists); ok {
					return nil
				}
			}
		}
	}

	return err
}
