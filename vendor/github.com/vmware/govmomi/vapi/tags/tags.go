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

package tags

import (
	"context"
	"fmt"
	"net/http"
	"strings"

	"github.com/vmware/govmomi/vapi/internal"
	"github.com/vmware/govmomi/vapi/rest"
)

// Manager extends rest.Client, adding tag related methods.
type Manager struct {
	*rest.Client
}

// NewManager creates a new Manager instance with the given client.
func NewManager(client *rest.Client) *Manager {
	return &Manager{
		Client: client,
	}
}

// isName returns true if the id is not a urn.
func isName(id string) bool {
	return !strings.HasPrefix(id, "urn:")
}

// Tag provides methods to create, read, update, delete, and enumerate tags.
type Tag struct {
	ID          string   `json:"id,omitempty"`
	Description string   `json:"description,omitempty"`
	Name        string   `json:"name,omitempty"`
	CategoryID  string   `json:"category_id,omitempty"`
	UsedBy      []string `json:"used_by,omitempty"`
}

// Patch merges updates from the given src.
func (t *Tag) Patch(src *Tag) {
	if src.Name != "" {
		t.Name = src.Name
	}
	if src.Description != "" {
		t.Description = src.Description
	}
	if src.CategoryID != "" {
		t.CategoryID = src.CategoryID
	}
}

// CreateTag creates a new tag with the given Name, Description and CategoryID.
func (c *Manager) CreateTag(ctx context.Context, tag *Tag) (string, error) {
	// create avoids the annoyance of CreateTag requiring a "description" key to be included in the request,
	// even though the field value can be empty.
	type create struct {
		Name        string `json:"name"`
		Description string `json:"description"`
		CategoryID  string `json:"category_id"`
	}
	spec := struct {
		Tag create `json:"create_spec"`
	}{
		Tag: create{
			Name:        tag.Name,
			Description: tag.Description,
			CategoryID:  tag.CategoryID,
		},
	}
	if isName(tag.CategoryID) {
		cat, err := c.GetCategory(ctx, tag.CategoryID)
		if err != nil {
			return "", err
		}
		spec.Tag.CategoryID = cat.ID
	}
	url := internal.URL(c, internal.TagPath)
	var res string
	return res, c.Do(ctx, url.Request(http.MethodPost, spec), &res)
}

// UpdateTag can update one or both of the tag Description and Name fields.
func (c *Manager) UpdateTag(ctx context.Context, tag *Tag) error {
	spec := struct {
		Tag Tag `json:"update_spec"`
	}{
		Tag: Tag{
			Name:        tag.Name,
			Description: tag.Description,
		},
	}
	url := internal.URL(c, internal.TagPath).WithID(tag.ID)
	return c.Do(ctx, url.Request(http.MethodPatch, spec), nil)
}

// DeleteTag deletes an existing tag.
func (c *Manager) DeleteTag(ctx context.Context, tag *Tag) error {
	url := internal.URL(c, internal.TagPath).WithID(tag.ID)
	return c.Do(ctx, url.Request(http.MethodDelete), nil)
}

// GetTag fetches the tag information for the given identifier.
// The id parameter can be a Tag ID or Tag Name.
func (c *Manager) GetTag(ctx context.Context, id string) (*Tag, error) {
	if isName(id) {
		tags, err := c.GetTags(ctx)
		if err != nil {
			return nil, err
		}

		for i := range tags {
			if tags[i].Name == id {
				return &tags[i], nil
			}
		}
	}

	url := internal.URL(c, internal.TagPath).WithID(id)
	var res Tag
	return &res, c.Do(ctx, url.Request(http.MethodGet), &res)

}

// ListTags returns all tag IDs in the system.
func (c *Manager) ListTags(ctx context.Context) ([]string, error) {
	url := internal.URL(c, internal.TagPath)
	var res []string
	return res, c.Do(ctx, url.Request(http.MethodGet), &res)
}

// GetTags fetches an array of tag information in the system.
func (c *Manager) GetTags(ctx context.Context) ([]Tag, error) {
	ids, err := c.ListTags(ctx)
	if err != nil {
		return nil, fmt.Errorf("get tags failed for: %s", err)
	}

	var tags []Tag
	for _, id := range ids {
		tag, err := c.GetTag(ctx, id)
		if err != nil {
			return nil, fmt.Errorf("get category %s failed for %s", id, err)
		}

		tags = append(tags, *tag)

	}
	return tags, nil
}

// The id parameter can be a Category ID or Category Name.
func (c *Manager) ListTagsForCategory(ctx context.Context, id string) ([]string, error) {
	if isName(id) {
		cat, err := c.GetCategory(ctx, id)
		if err != nil {
			return nil, err
		}
		id = cat.ID
	}

	body := struct {
		ID string `json:"category_id"`
	}{id}
	url := internal.URL(c, internal.TagPath).WithID(id).WithAction("list-tags-for-category")
	var res []string
	return res, c.Do(ctx, url.Request(http.MethodPost, body), &res)
}

// The id parameter can be a Category ID or Category Name.
func (c *Manager) GetTagsForCategory(ctx context.Context, id string) ([]Tag, error) {
	ids, err := c.ListTagsForCategory(ctx, id)
	if err != nil {
		return nil, err
	}

	var tags []Tag
	for _, id := range ids {
		tag, err := c.GetTag(ctx, id)
		if err != nil {
			return nil, fmt.Errorf("get tag %s: %s", id, err)
		}

		tags = append(tags, *tag)
	}
	return tags, nil
}
