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

package simulator

import (
	"io/ioutil"
	"log"
	"os"
	"path"
	"strings"

	"github.com/vmware/govmomi/vim25/methods"
	"github.com/vmware/govmomi/vim25/mo"
	"github.com/vmware/govmomi/vim25/soap"
	"github.com/vmware/govmomi/vim25/types"
)

type HostDatastoreBrowser struct {
	mo.HostDatastoreBrowser
}

type searchDatastore struct {
	*HostDatastoreBrowser

	DatastorePath string
	SearchSpec    *types.HostDatastoreBrowserSearchSpec

	res []types.HostDatastoreBrowserSearchResults

	recurse bool
}

func (s *searchDatastore) addFile(file os.FileInfo, res *types.HostDatastoreBrowserSearchResults) {
	details := s.SearchSpec.Details
	if details == nil {
		details = new(types.FileQueryFlags)
	}

	name := file.Name()

	info := types.FileInfo{
		Path: name,
	}

	var finfo types.BaseFileInfo = &info

	if details.FileSize {
		info.FileSize = file.Size()
	}

	if details.Modification {
		mtime := file.ModTime()
		info.Modification = &mtime
	}

	if isTrue(details.FileOwner) {
		// Assume for now this process created all files in the datastore
		user := os.Getenv("USER")

		info.Owner = user
	}

	if file.IsDir() {
		finfo = &types.FolderFileInfo{FileInfo: info}
	} else if details.FileType {
		switch path.Ext(name) {
		case ".img":
			finfo = &types.FloppyImageFileInfo{FileInfo: info}
		case ".iso":
			finfo = &types.IsoImageFileInfo{FileInfo: info}
		case ".log":
			finfo = &types.VmLogFileInfo{FileInfo: info}
		case ".nvram":
			finfo = &types.VmNvramFileInfo{FileInfo: info}
		case ".vmdk":
			// TODO: lookup device to set other fields
			finfo = &types.VmDiskFileInfo{FileInfo: info}
		case ".vmx":
			finfo = &types.VmConfigFileInfo{FileInfo: info}
		}
	}

	res.File = append(res.File, finfo)
}

func (s *searchDatastore) queryMatch(file os.FileInfo) bool {
	if len(s.SearchSpec.Query) == 0 {
		return true
	}

	name := file.Name()
	ext := path.Ext(name)

	for _, q := range s.SearchSpec.Query {
		switch q.(type) {
		case *types.FileQuery:
			return true
		case *types.FolderFileQuery:
			if file.IsDir() {
				return true
			}
		case *types.FloppyImageFileQuery:
			if ext == ".img" {
				return true
			}
		case *types.IsoImageFileQuery:
			if ext == ".iso" {
				return true
			}
		case *types.VmConfigFileQuery:
			if ext == ".vmx" {
				// TODO: check Filter and Details fields
				return true
			}
		case *types.VmDiskFileQuery:
			if ext == ".vmdk" {
				if strings.HasSuffix(name, "-flat.vmdk") {
					// only matches the descriptor, not the backing file(s)
					return false
				}
				// TODO: check Filter and Details fields
				return true
			}
		case *types.VmLogFileQuery:
			if ext == ".log" {
				return strings.HasPrefix(name, "vmware")
			}
		case *types.VmNvramFileQuery:
			if ext == ".nvram" {
				return true
			}
		case *types.VmSnapshotFileQuery:
			if ext == ".vmsn" {
				return true
			}
		}
	}

	return false
}

func (s *searchDatastore) search(ds *types.ManagedObjectReference, folder string, dir string) error {
	files, err := ioutil.ReadDir(dir)
	if err != nil {
		log.Printf("search %s: %s", dir, err)
		return err
	}

	res := types.HostDatastoreBrowserSearchResults{
		Datastore:  ds,
		FolderPath: folder,
	}

	for _, file := range files {
		name := file.Name()

		if s.queryMatch(file) {
			for _, m := range s.SearchSpec.MatchPattern {
				if ok, _ := path.Match(m, name); ok {
					s.addFile(file, &res)
					break
				}
			}
		}

		if s.recurse && file.IsDir() {
			_ = s.search(ds, path.Join(folder, name), path.Join(dir, name))
		}
	}

	s.res = append(s.res, res)

	return nil
}

func (s *searchDatastore) Run(task *Task) (types.AnyType, types.BaseMethodFault) {
	p, fault := parseDatastorePath(s.DatastorePath)
	if fault != nil {
		return nil, fault
	}

	ref := Map.FindByName(p.Datastore, s.Datastore)
	if ref == nil {
		return nil, &types.InvalidDatastore{Name: p.Datastore}
	}

	ds := ref.(*Datastore)
	task.Info.Entity = &ds.Self // TODO: CreateTask() should require mo.Entity, rather than mo.Reference
	task.Info.EntityName = ds.Name

	dir := path.Join(ds.Info.GetDatastoreInfo().Url, p.Path)

	err := s.search(&ds.Self, s.DatastorePath, dir)
	if err != nil {
		ff := types.FileFault{
			File: p.Path,
		}

		if os.IsNotExist(err) {
			return nil, &types.FileNotFound{FileFault: ff}
		}

		return nil, &types.InvalidArgument{InvalidProperty: p.Path}
	}

	if s.recurse {
		return types.ArrayOfHostDatastoreBrowserSearchResults{
			HostDatastoreBrowserSearchResults: s.res,
		}, nil
	}

	return s.res[0], nil
}

func (b *HostDatastoreBrowser) SearchDatastoreTask(s *types.SearchDatastore_Task) soap.HasFault {
	task := NewTask(&searchDatastore{
		HostDatastoreBrowser: b,
		DatastorePath:        s.DatastorePath,
		SearchSpec:           s.SearchSpec,
	})

	return &methods.SearchDatastore_TaskBody{
		Res: &types.SearchDatastore_TaskResponse{
			Returnval: task.Run(),
		},
	}
}

func (b *HostDatastoreBrowser) SearchDatastoreSubFoldersTask(s *types.SearchDatastoreSubFolders_Task) soap.HasFault {
	task := NewTask(&searchDatastore{
		HostDatastoreBrowser: b,
		DatastorePath:        s.DatastorePath,
		SearchSpec:           s.SearchSpec,
		recurse:              true,
	})

	return &methods.SearchDatastoreSubFolders_TaskBody{
		Res: &types.SearchDatastoreSubFolders_TaskResponse{
			Returnval: task.Run(),
		},
	}
}
