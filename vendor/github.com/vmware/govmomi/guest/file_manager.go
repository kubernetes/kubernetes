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

package guest

import (
	"context"

	"github.com/vmware/govmomi/vim25"
	"github.com/vmware/govmomi/vim25/methods"
	"github.com/vmware/govmomi/vim25/types"
)

type FileManager struct {
	types.ManagedObjectReference

	vm types.ManagedObjectReference

	c *vim25.Client
}

func (m FileManager) Reference() types.ManagedObjectReference {
	return m.ManagedObjectReference
}

func (m FileManager) ChangeFileAttributes(ctx context.Context, auth types.BaseGuestAuthentication, guestFilePath string, fileAttributes types.BaseGuestFileAttributes) error {
	req := types.ChangeFileAttributesInGuest{
		This:           m.Reference(),
		Vm:             m.vm,
		Auth:           auth,
		GuestFilePath:  guestFilePath,
		FileAttributes: fileAttributes,
	}

	_, err := methods.ChangeFileAttributesInGuest(ctx, m.c, &req)
	return err
}

func (m FileManager) CreateTemporaryDirectory(ctx context.Context, auth types.BaseGuestAuthentication, prefix, suffix string) (string, error) {
	req := types.CreateTemporaryDirectoryInGuest{
		This:   m.Reference(),
		Vm:     m.vm,
		Auth:   auth,
		Prefix: prefix,
		Suffix: suffix,
	}

	res, err := methods.CreateTemporaryDirectoryInGuest(ctx, m.c, &req)
	if err != nil {
		return "", err
	}

	return res.Returnval, nil
}

func (m FileManager) CreateTemporaryFile(ctx context.Context, auth types.BaseGuestAuthentication, prefix, suffix string) (string, error) {
	req := types.CreateTemporaryFileInGuest{
		This:   m.Reference(),
		Vm:     m.vm,
		Auth:   auth,
		Prefix: prefix,
		Suffix: suffix,
	}

	res, err := methods.CreateTemporaryFileInGuest(ctx, m.c, &req)
	if err != nil {
		return "", err
	}

	return res.Returnval, nil
}

func (m FileManager) DeleteDirectory(ctx context.Context, auth types.BaseGuestAuthentication, directoryPath string, recursive bool) error {
	req := types.DeleteDirectoryInGuest{
		This:          m.Reference(),
		Vm:            m.vm,
		Auth:          auth,
		DirectoryPath: directoryPath,
		Recursive:     recursive,
	}

	_, err := methods.DeleteDirectoryInGuest(ctx, m.c, &req)
	return err
}

func (m FileManager) DeleteFile(ctx context.Context, auth types.BaseGuestAuthentication, filePath string) error {
	req := types.DeleteFileInGuest{
		This:     m.Reference(),
		Vm:       m.vm,
		Auth:     auth,
		FilePath: filePath,
	}

	_, err := methods.DeleteFileInGuest(ctx, m.c, &req)
	return err
}

func (m FileManager) InitiateFileTransferFromGuest(ctx context.Context, auth types.BaseGuestAuthentication, guestFilePath string) (*types.FileTransferInformation, error) {
	req := types.InitiateFileTransferFromGuest{
		This:          m.Reference(),
		Vm:            m.vm,
		Auth:          auth,
		GuestFilePath: guestFilePath,
	}

	res, err := methods.InitiateFileTransferFromGuest(ctx, m.c, &req)
	if err != nil {
		return nil, err
	}

	return &res.Returnval, nil
}

func (m FileManager) InitiateFileTransferToGuest(ctx context.Context, auth types.BaseGuestAuthentication, guestFilePath string, fileAttributes types.BaseGuestFileAttributes, fileSize int64, overwrite bool) (string, error) {
	req := types.InitiateFileTransferToGuest{
		This:           m.Reference(),
		Vm:             m.vm,
		Auth:           auth,
		GuestFilePath:  guestFilePath,
		FileAttributes: fileAttributes,
		FileSize:       fileSize,
		Overwrite:      overwrite,
	}

	res, err := methods.InitiateFileTransferToGuest(ctx, m.c, &req)
	if err != nil {
		return "", err
	}

	return res.Returnval, nil
}

func (m FileManager) ListFiles(ctx context.Context, auth types.BaseGuestAuthentication, filePath string, index int32, maxResults int32, matchPattern string) (*types.GuestListFileInfo, error) {
	req := types.ListFilesInGuest{
		This:         m.Reference(),
		Vm:           m.vm,
		Auth:         auth,
		FilePath:     filePath,
		Index:        index,
		MaxResults:   maxResults,
		MatchPattern: matchPattern,
	}

	res, err := methods.ListFilesInGuest(ctx, m.c, &req)
	if err != nil {
		return nil, err
	}

	return &res.Returnval, nil
}

func (m FileManager) MakeDirectory(ctx context.Context, auth types.BaseGuestAuthentication, directoryPath string, createParentDirectories bool) error {
	req := types.MakeDirectoryInGuest{
		This:                    m.Reference(),
		Vm:                      m.vm,
		Auth:                    auth,
		DirectoryPath:           directoryPath,
		CreateParentDirectories: createParentDirectories,
	}

	_, err := methods.MakeDirectoryInGuest(ctx, m.c, &req)
	return err
}

func (m FileManager) MoveDirectory(ctx context.Context, auth types.BaseGuestAuthentication, srcDirectoryPath string, dstDirectoryPath string) error {
	req := types.MoveDirectoryInGuest{
		This:             m.Reference(),
		Vm:               m.vm,
		Auth:             auth,
		SrcDirectoryPath: srcDirectoryPath,
		DstDirectoryPath: dstDirectoryPath,
	}

	_, err := methods.MoveDirectoryInGuest(ctx, m.c, &req)
	return err
}

func (m FileManager) MoveFile(ctx context.Context, auth types.BaseGuestAuthentication, srcFilePath string, dstFilePath string, overwrite bool) error {
	req := types.MoveFileInGuest{
		This:        m.Reference(),
		Vm:          m.vm,
		Auth:        auth,
		SrcFilePath: srcFilePath,
		DstFilePath: dstFilePath,
		Overwrite:   overwrite,
	}

	_, err := methods.MoveFileInGuest(ctx, m.c, &req)
	return err
}
