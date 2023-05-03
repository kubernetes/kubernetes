/*
Copyright (c) 2018 VMware, Inc. All Rights Reserved.

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

package library

import (
	"context"
	"net/http"

	"github.com/vmware/govmomi/vapi/internal"
	"github.com/vmware/govmomi/vapi/rest"
)

// DownloadFile is the specification for the downloadsession
// operations file:add, file:get, and file:list.
type DownloadFile struct {
	BytesTransferred int64                    `json:"bytes_transferred"`
	Checksum         *Checksum                `json:"checksum_info,omitempty"`
	DownloadEndpoint *TransferEndpoint        `json:"download_endpoint,omitempty"`
	ErrorMessage     *rest.LocalizableMessage `json:"error_message,omitempty"`
	Name             string                   `json:"name"`
	Size             int64                    `json:"size,omitempty"`
	Status           string                   `json:"status"`
}

// GetLibraryItemDownloadSessionFile retrieves information about a specific file that is a part of an download session.
func (c *Manager) GetLibraryItemDownloadSessionFile(ctx context.Context, sessionID string, name string) (*DownloadFile, error) {
	url := c.Resource(internal.LibraryItemDownloadSessionFile).WithID(sessionID).WithAction("get")
	spec := struct {
		Name string `json:"file_name"`
	}{name}
	var res DownloadFile
	err := c.Do(ctx, url.Request(http.MethodPost, spec), &res)
	if err != nil {
		return nil, err
	}
	if res.Status == "ERROR" {
		return nil, res.ErrorMessage
	}
	return &res, nil
}

// ListLibraryItemDownloadSessionFile retrieves information about a specific file that is a part of an download session.
func (c *Manager) ListLibraryItemDownloadSessionFile(ctx context.Context, sessionID string) ([]DownloadFile, error) {
	url := c.Resource(internal.LibraryItemDownloadSessionFile).WithParam("download_session_id", sessionID)
	var res []DownloadFile
	return res, c.Do(ctx, url.Request(http.MethodGet), &res)
}

// PrepareLibraryItemDownloadSessionFile retrieves information about a specific file that is a part of an download session.
func (c *Manager) PrepareLibraryItemDownloadSessionFile(ctx context.Context, sessionID string, name string) (*DownloadFile, error) {
	url := c.Resource(internal.LibraryItemDownloadSessionFile).WithID(sessionID).WithAction("prepare")
	spec := struct {
		Name string `json:"file_name"`
	}{name}
	var res DownloadFile
	return &res, c.Do(ctx, url.Request(http.MethodPost, spec), &res)
}
