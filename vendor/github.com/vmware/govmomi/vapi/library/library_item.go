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
	"fmt"
	"net/http"
	"time"

	"github.com/vmware/govmomi/vapi/internal"
)

const (
	ItemTypeISO  = "iso"
	ItemTypeOVF  = "ovf"
	ItemTypeVMTX = "vm-template"
)

// Item provides methods to create, read, update, delete, and enumerate library items.
type Item struct {
	Cached           bool       `json:"cached,omitempty"`
	ContentVersion   string     `json:"content_version,omitempty"`
	CreationTime     *time.Time `json:"creation_time,omitempty"`
	Description      string     `json:"description,omitempty"`
	ID               string     `json:"id,omitempty"`
	LastModifiedTime *time.Time `json:"last_modified_time,omitempty"`
	LastSyncTime     *time.Time `json:"last_sync_time,omitempty"`
	LibraryID        string     `json:"library_id,omitempty"`
	MetadataVersion  string     `json:"metadata_version,omitempty"`
	Name             string     `json:"name,omitempty"`
	Size             int64      `json:"size,omitempty"`
	SourceID         string     `json:"source_id,omitempty"`
	Type             string     `json:"type,omitempty"`
	Version          string     `json:"version,omitempty"`
}

// Patch merges updates from the given src.
func (i *Item) Patch(src *Item) {
	if src.Name != "" {
		i.Name = src.Name
	}
	if src.Description != "" {
		i.Description = src.Description
	}
	if src.Type != "" {
		i.Type = src.Type
	}
	if src.Version != "" {
		i.Version = src.Version
	}
}

// CreateLibraryItem creates a new library item
func (c *Manager) CreateLibraryItem(ctx context.Context, item Item) (string, error) {
	type createItemSpec struct {
		Name        string `json:"name"`
		Description string `json:"description"`
		LibraryID   string `json:"library_id,omitempty"`
		Type        string `json:"type"`
	}
	spec := struct {
		Item createItemSpec `json:"create_spec"`
	}{
		Item: createItemSpec{
			Name:        item.Name,
			Description: item.Description,
			LibraryID:   item.LibraryID,
			Type:        item.Type,
		},
	}
	url := c.Resource(internal.LibraryItemPath)
	var res string
	return res, c.Do(ctx, url.Request(http.MethodPost, spec), &res)
}

// CopyLibraryItem copies a library item
func (c *Manager) CopyLibraryItem(ctx context.Context, src *Item, dst Item) (string, error) {
	body := struct {
		Item `json:"destination_create_spec"`
	}{dst}
	url := c.Resource(internal.LibraryItemPath).WithID(src.ID).WithAction("copy")
	var res string
	return res, c.Do(ctx, url.Request(http.MethodPost, body), &res)
}

// SyncLibraryItem syncs a subscribed library item
func (c *Manager) SyncLibraryItem(ctx context.Context, item *Item, force bool) error {
	body := struct {
		Force bool `json:"force_sync_content"`
	}{force}
	url := c.Resource(internal.SubscribedLibraryItem).WithID(item.ID).WithAction("sync")
	return c.Do(ctx, url.Request(http.MethodPost, body), nil)
}

// PublishLibraryItem publishes a library item to specified subscriptions.
// If no subscriptions are specified, then publishes the library item to all subscriptions.
func (c *Manager) PublishLibraryItem(ctx context.Context, item *Item, force bool, subscriptions []string) error {
	body := internal.SubscriptionItemDestinationSpec{
		Force: force,
	}
	for i := range subscriptions {
		body.Subscriptions = append(body.Subscriptions, internal.SubscriptionDestination{ID: subscriptions[i]})
	}
	url := c.Resource(internal.LibraryItemPath).WithID(item.ID).WithAction("publish")
	return c.Do(ctx, url.Request(http.MethodPost, body), nil)
}

// DeleteLibraryItem deletes an existing library item.
func (c *Manager) DeleteLibraryItem(ctx context.Context, item *Item) error {
	url := c.Resource(internal.LibraryItemPath).WithID(item.ID)
	return c.Do(ctx, url.Request(http.MethodDelete), nil)
}

// ListLibraryItems returns a list of all items in a content library.
func (c *Manager) ListLibraryItems(ctx context.Context, id string) ([]string, error) {
	url := c.Resource(internal.LibraryItemPath).WithParam("library_id", id)
	var res []string
	return res, c.Do(ctx, url.Request(http.MethodGet), &res)
}

// GetLibraryItem returns information on a library item for the given ID.
func (c *Manager) GetLibraryItem(ctx context.Context, id string) (*Item, error) {
	url := c.Resource(internal.LibraryItemPath).WithID(id)
	var res Item
	return &res, c.Do(ctx, url.Request(http.MethodGet), &res)
}

// GetLibraryItems returns a list of all the library items for the specified library.
func (c *Manager) GetLibraryItems(ctx context.Context, libraryID string) ([]Item, error) {
	ids, err := c.ListLibraryItems(ctx, libraryID)
	if err != nil {
		return nil, fmt.Errorf("get library items failed for: %s", err)
	}
	var items []Item
	for _, id := range ids {
		item, err := c.GetLibraryItem(ctx, id)
		if err != nil {
			return nil, fmt.Errorf("get library item for %s failed for %s", id, err)
		}
		items = append(items, *item)
	}
	return items, nil
}

// FindItem is the search criteria for finding library items.
type FindItem struct {
	Cached    *bool  `json:"cached,omitempty"`
	LibraryID string `json:"library_id,omitempty"`
	Name      string `json:"name,omitempty"`
	SourceID  string `json:"source_id,omitempty"`
	Type      string `json:"type,omitempty"`
}

// FindLibraryItems returns the IDs of all the library items that match the
// search criteria.
func (c *Manager) FindLibraryItems(
	ctx context.Context, search FindItem) ([]string, error) {

	url := c.Resource(internal.LibraryItemPath).WithAction("find")
	spec := struct {
		Spec FindItem `json:"spec"`
	}{search}
	var res []string
	return res, c.Do(ctx, url.Request(http.MethodPost, spec), &res)
}
