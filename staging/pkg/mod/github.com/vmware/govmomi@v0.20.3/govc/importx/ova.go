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

package importx

import (
	"context"
	"flag"

	"github.com/vmware/govmomi/govc/cli"
	"github.com/vmware/govmomi/object"
	"github.com/vmware/govmomi/vim25/types"
)

type ova struct {
	*ovfx
}

func init() {
	cli.Register("import.ova", &ova{&ovfx{}})
}

func (cmd *ova) Usage() string {
	return "PATH_TO_OVA"
}

func (cmd *ova) Run(ctx context.Context, f *flag.FlagSet) error {
	fpath, err := cmd.Prepare(f)
	if err != nil {
		return err
	}

	archive := &TapeArchive{path: fpath}
	archive.Client = cmd.Client

	cmd.Archive = archive

	moref, err := cmd.Import(fpath)
	if err != nil {
		return err
	}

	vm := object.NewVirtualMachine(cmd.Client, *moref)
	return cmd.Deploy(vm)
}

func (cmd *ova) Import(fpath string) (*types.ManagedObjectReference, error) {
	ovf := "*.ovf"
	return cmd.ovfx.Import(ovf)
}
