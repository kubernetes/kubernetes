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

	"github.com/vmware/govmomi/vapi/internal"
)

// Category provides methods to create, read, update, delete, and enumerate categories.
type Category struct {
	ID              string   `json:"id,omitempty"`
	Name            string   `json:"name,omitempty"`
	Description     string   `json:"description,omitempty"`
	Cardinality     string   `json:"cardinality,omitempty"`
	AssociableTypes []string `json:"associable_types,omitempty"`
	UsedBy          []string `json:"used_by,omitempty"`
}

func (c *Category) hasType(kind string) bool {
	for _, k := range c.AssociableTypes {
		if kind == k {
			return true
		}
	}
	return false
}

// Patch merges Category changes from the given src.
// AssociableTypes can only be appended to and cannot shrink.
func (c *Category) Patch(src *Category) {
	if src.Name != "" {
		c.Name = src.Name
	}
	if src.Description != "" {
		c.Description = src.Description
	}
	if src.Cardinality != "" {
		c.Cardinality = src.Cardinality
	}
	// Note that in order to append to AssociableTypes any existing types must be included in their original order.
	for _, kind := range src.AssociableTypes {
		if !c.hasType(kind) {
			c.AssociableTypes = append(c.AssociableTypes, kind)
		}
	}
}

// CreateCategory creates a new category and returns the category ID.
func (c *Manager) CreateCategory(ctx context.Context, category *Category) (string, error) {
	// create avoids the annoyance of CreateTag requiring field keys to be included in the request,
	// even though the field value can be empty.
	type create struct {
		Name            string   `json:"name"`
		Description     string   `json:"description"`
		Cardinality     string   `json:"cardinality"`
		AssociableTypes []string `json:"associable_types"`
	}
	spec := struct {
		Category create `json:"create_spec"`
	}{
		Category: create{
			Name:            category.Name,
			Description:     category.Description,
			Cardinality:     category.Cardinality,
			AssociableTypes: category.AssociableTypes,
		},
	}
	if spec.Category.AssociableTypes == nil {
		// otherwise create fails with invalid_argument
		spec.Category.AssociableTypes = []string{}
	}
	url := internal.URL(c, internal.CategoryPath)
	var res string
	return res, c.Do(ctx, url.Request(http.MethodPost, spec), &res)
}

// UpdateCategory can update one or more of the AssociableTypes, Cardinality, Description and Name fields.
func (c *Manager) UpdateCategory(ctx context.Context, category *Category) error {
	spec := struct {
		Category Category `json:"update_spec"`
	}{
		Category: Category{
			AssociableTypes: category.AssociableTypes,
			Cardinality:     category.Cardinality,
			Description:     category.Description,
			Name:            category.Name,
		},
	}
	url := internal.URL(c, internal.CategoryPath).WithID(category.ID)
	return c.Do(ctx, url.Request(http.MethodPatch, spec), nil)
}

// DeleteCategory deletes an existing category.
func (c *Manager) DeleteCategory(ctx context.Context, category *Category) error {
	url := internal.URL(c, internal.CategoryPath).WithID(category.ID)
	return c.Do(ctx, url.Request(http.MethodDelete), nil)
}

// GetCategory fetches the category information for the given identifier.
// The id parameter can be a Category ID or Category Name.
func (c *Manager) GetCategory(ctx context.Context, id string) (*Category, error) {
	if isName(id) {
		cat, err := c.GetCategories(ctx)
		if err != nil {
			return nil, err
		}

		for i := range cat {
			if cat[i].Name == id {
				return &cat[i], nil
			}
		}
	}
	url := internal.URL(c, internal.CategoryPath).WithID(id)
	var res Category
	return &res, c.Do(ctx, url.Request(http.MethodGet), &res)
}

// ListCategories returns all category IDs in the system.
func (c *Manager) ListCategories(ctx context.Context) ([]string, error) {
	url := internal.URL(c, internal.CategoryPath)
	var res []string
	return res, c.Do(ctx, url.Request(http.MethodGet), &res)
}

// GetCategories fetches an array of category information in the system.
func (c *Manager) GetCategories(ctx context.Context) ([]Category, error) {
	ids, err := c.ListCategories(ctx)
	if err != nil {
		return nil, fmt.Errorf("list categories: %s", err)
	}

	var categories []Category
	for _, id := range ids {
		category, err := c.GetCategory(ctx, id)
		if err != nil {
			return nil, fmt.Errorf("get category %s: %s", id, err)
		}

		categories = append(categories, *category)

	}
	return categories, nil
}
