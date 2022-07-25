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

package object

import (
	"context"

	"github.com/vmware/govmomi/vim25"
	"github.com/vmware/govmomi/vim25/methods"
	"github.com/vmware/govmomi/vim25/types"
)

type FileManager struct {
	Common
}

func NewFileManager(c *vim25.Client) *FileManager {
	f := FileManager{
		Common: NewCommon(c, *c.ServiceContent.FileManager),
	}

	return &f
}

func (f FileManager) CopyDatastoreFile(ctx context.Context, sourceName string, sourceDatacenter *Datacenter, destinationName string, destinationDatacenter *Datacenter, force bool) (*Task, error) {
	req := types.CopyDatastoreFile_Task{
		This:            f.Reference(),
		SourceName:      sourceName,
		DestinationName: destinationName,
		Force:           types.NewBool(force),
	}

	if sourceDatacenter != nil {
		ref := sourceDatacenter.Reference()
		req.SourceDatacenter = &ref
	}

	if destinationDatacenter != nil {
		ref := destinationDatacenter.Reference()
		req.DestinationDatacenter = &ref
	}

	res, err := methods.CopyDatastoreFile_Task(ctx, f.c, &req)
	if err != nil {
		return nil, err
	}

	return NewTask(f.c, res.Returnval), nil
}

// DeleteDatastoreFile deletes the specified file or folder from the datastore.
func (f FileManager) DeleteDatastoreFile(ctx context.Context, name string, dc *Datacenter) (*Task, error) {
	req := types.DeleteDatastoreFile_Task{
		This: f.Reference(),
		Name: name,
	}

	if dc != nil {
		ref := dc.Reference()
		req.Datacenter = &ref
	}

	res, err := methods.DeleteDatastoreFile_Task(ctx, f.c, &req)
	if err != nil {
		return nil, err
	}

	return NewTask(f.c, res.Returnval), nil
}

// MakeDirectory creates a folder using the specified name.
func (f FileManager) MakeDirectory(ctx context.Context, name string, dc *Datacenter, createParentDirectories bool) error {
	req := types.MakeDirectory{
		This:                    f.Reference(),
		Name:                    name,
		CreateParentDirectories: types.NewBool(createParentDirectories),
	}

	if dc != nil {
		ref := dc.Reference()
		req.Datacenter = &ref
	}

	_, err := methods.MakeDirectory(ctx, f.c, &req)
	return err
}

func (f FileManager) MoveDatastoreFile(ctx context.Context, sourceName string, sourceDatacenter *Datacenter, destinationName string, destinationDatacenter *Datacenter, force bool) (*Task, error) {
	req := types.MoveDatastoreFile_Task{
		This:            f.Reference(),
		SourceName:      sourceName,
		DestinationName: destinationName,
		Force:           types.NewBool(force),
	}

	if sourceDatacenter != nil {
		ref := sourceDatacenter.Reference()
		req.SourceDatacenter = &ref
	}

	if destinationDatacenter != nil {
		ref := destinationDatacenter.Reference()
		req.DestinationDatacenter = &ref
	}

	res, err := methods.MoveDatastoreFile_Task(ctx, f.c, &req)
	if err != nil {
		return nil, err
	}

	return NewTask(f.c, res.Returnval), nil
}
