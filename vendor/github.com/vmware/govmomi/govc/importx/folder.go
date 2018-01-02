/*
Copyright (c) 2015 VMware, Inc. All Rights Reserved.

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
	"errors"
	"flag"

	"github.com/vmware/govmomi/govc/flags"
	"github.com/vmware/govmomi/object"
)

type FolderFlag struct {
	*flags.DatacenterFlag

	folder string
}

func newFolderFlag(ctx context.Context) (*FolderFlag, context.Context) {
	f := &FolderFlag{}
	f.DatacenterFlag, ctx = flags.NewDatacenterFlag(ctx)
	return f, ctx
}

func (flag *FolderFlag) Register(ctx context.Context, f *flag.FlagSet) {
	flag.DatacenterFlag.Register(ctx, f)

	f.StringVar(&flag.folder, "folder", "", "Path to folder to add the VM to")
}

func (flag *FolderFlag) Process(ctx context.Context) error {
	return flag.DatacenterFlag.Process(ctx)
}

func (flag *FolderFlag) Folder() (*object.Folder, error) {
	ctx := context.TODO()
	if len(flag.folder) == 0 {
		dc, err := flag.Datacenter()
		if err != nil {
			return nil, err
		}
		folders, err := dc.Folders(ctx)
		if err != nil {
			return nil, err
		}
		return folders.VmFolder, nil
	}

	finder, err := flag.Finder()
	if err != nil {
		return nil, err
	}

	mo, err := finder.ManagedObjectList(ctx, flag.folder)
	if err != nil {
		return nil, err
	}
	if len(mo) == 0 {
		return nil, errors.New("folder argument does not resolve to object")
	}
	if len(mo) > 1 {
		return nil, errors.New("folder argument resolves to more than one object")
	}

	ref := mo[0].Object.Reference()
	if ref.Type != "Folder" {
		return nil, errors.New("folder argument does not resolve to folder")
	}

	c, err := flag.Client()
	if err != nil {
		return nil, err
	}

	return object.NewFolder(c, ref), nil
}
