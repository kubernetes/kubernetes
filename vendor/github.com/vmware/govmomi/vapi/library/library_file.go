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
)

// Checksum provides checksum information on library item files.
type Checksum struct {
	Algorithm string `json:"algorithm,omitempty"`
	Checksum  string `json:"checksum"`
}

// File provides methods to get information on library item files.
type File struct {
	Cached   *bool     `json:"cached,omitempty"`
	Checksum *Checksum `json:"checksum_info,omitempty"`
	Name     string    `json:"name,omitempty"`
	Size     *int64    `json:"size,omitempty"`
	Version  string    `json:"version,omitempty"`
}

// ListLibraryItemFiles returns a list of all the files for a library item.
func (c *Manager) ListLibraryItemFiles(ctx context.Context, id string) ([]File, error) {
	url := c.Resource(internal.LibraryItemFilePath).WithParam("library_item_id", id)
	var res []File
	return res, c.Do(ctx, url.Request(http.MethodGet), &res)
}

// GetLibraryItemFile returns a file with the provided name for a library item.
func (c *Manager) GetLibraryItemFile(ctx context.Context, id, fileName string) (*File, error) {
	url := c.Resource(internal.LibraryItemFilePath).WithID(id).WithAction("get")
	spec := struct {
		Name string `json:"name"`
	}{fileName}
	var res File
	return &res, c.Do(ctx, url.Request(http.MethodPost, spec), &res)
}
