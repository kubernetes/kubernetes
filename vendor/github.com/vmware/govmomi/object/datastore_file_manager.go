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

package object

import (
	"bufio"
	"bytes"
	"context"
	"fmt"
	"io"
	"log"
	"path"
	"strings"

	"github.com/vmware/govmomi/vim25/soap"
)

// DatastoreFileManager combines FileManager and VirtualDiskManager to manage files on a Datastore
type DatastoreFileManager struct {
	Datacenter         *Datacenter
	Datastore          *Datastore
	FileManager        *FileManager
	VirtualDiskManager *VirtualDiskManager

	Force bool
}

// NewFileManager creates a new instance of DatastoreFileManager
func (d Datastore) NewFileManager(dc *Datacenter, force bool) *DatastoreFileManager {
	c := d.Client()

	m := &DatastoreFileManager{
		Datacenter:         dc,
		Datastore:          &d,
		FileManager:        NewFileManager(c),
		VirtualDiskManager: NewVirtualDiskManager(c),
		Force:              force,
	}

	return m
}

// Delete dispatches to the appropriate Delete method based on file name extension
func (m *DatastoreFileManager) Delete(ctx context.Context, name string) error {
	switch path.Ext(name) {
	case ".vmdk":
		return m.DeleteVirtualDisk(ctx, name)
	default:
		return m.DeleteFile(ctx, name)
	}
}

// DeleteFile calls FileManager.DeleteDatastoreFile
func (m *DatastoreFileManager) DeleteFile(ctx context.Context, name string) error {
	p := m.Path(name)

	task, err := m.FileManager.DeleteDatastoreFile(ctx, p.String(), m.Datacenter)
	if err != nil {
		return err
	}

	return task.Wait(ctx)
}

// DeleteVirtualDisk calls VirtualDiskManager.DeleteVirtualDisk
// Regardless of the Datastore type, DeleteVirtualDisk will fail if 'ddb.deletable=false',
// so if Force=true this method attempts to set 'ddb.deletable=true' before starting the delete task.
func (m *DatastoreFileManager) DeleteVirtualDisk(ctx context.Context, name string) error {
	p := m.Path(name)

	var merr error

	if m.Force {
		merr = m.markDiskAsDeletable(ctx, p)
	}

	task, err := m.VirtualDiskManager.DeleteVirtualDisk(ctx, p.String(), m.Datacenter)
	if err != nil {
		log.Printf("markDiskAsDeletable(%s): %s", p, merr)
		return err
	}

	return task.Wait(ctx)
}

// Move dispatches to the appropriate Move method based on file name extension
func (m *DatastoreFileManager) Move(ctx context.Context, src string, dst string) error {
	srcp := m.Path(src)
	dstp := m.Path(dst)

	f := m.FileManager.MoveDatastoreFile

	if srcp.IsVMDK() {
		f = m.VirtualDiskManager.MoveVirtualDisk
	}

	task, err := f(ctx, srcp.String(), m.Datacenter, dstp.String(), m.Datacenter, m.Force)
	if err != nil {
		return err
	}

	return task.Wait(ctx)
}

// Path converts path name to a DatastorePath
func (m *DatastoreFileManager) Path(name string) *DatastorePath {
	var p DatastorePath

	if !p.FromString(name) {
		p.Path = name
		p.Datastore = m.Datastore.Name()
	}

	return &p
}

func (m *DatastoreFileManager) markDiskAsDeletable(ctx context.Context, path *DatastorePath) error {
	r, _, err := m.Datastore.Download(ctx, path.Path, &soap.DefaultDownload)
	if err != nil {
		return err
	}

	defer r.Close()

	hasFlag := false
	buf := new(bytes.Buffer)

	s := bufio.NewScanner(&io.LimitedReader{R: r, N: 2048}) // should be only a few hundred bytes, limit to be sure

	for s.Scan() {
		line := s.Text()
		if strings.HasPrefix(line, "ddb.deletable") {
			hasFlag = true
			continue
		}

		fmt.Fprintln(buf, line)
	}

	if err := s.Err(); err != nil {
		return err // any error other than EOF
	}

	if !hasFlag {
		return nil // already deletable, so leave as-is
	}

	// rewrite the .vmdk with ddb.deletable flag removed (the default is true)
	return m.Datastore.Upload(ctx, buf, path.Path, &soap.DefaultUpload)
}
