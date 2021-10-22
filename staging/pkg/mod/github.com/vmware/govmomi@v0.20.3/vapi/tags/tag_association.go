/*
Copyright (c) 2018 VMware, Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

vUnless required by applicable law or agreed to in writing, software
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
	"github.com/vmware/govmomi/vim25/mo"
)

func (c *Manager) tagID(ctx context.Context, id string) (string, error) {
	if isName(id) {
		tag, err := c.GetTag(ctx, id)
		if err != nil {
			return "", err
		}
		return tag.ID, nil
	}
	return id, nil
}

// AttachTag attaches a tag ID to a managed object.
func (c *Manager) AttachTag(ctx context.Context, tagID string, ref mo.Reference) error {
	id, err := c.tagID(ctx, tagID)
	if err != nil {
		return err
	}
	spec := internal.NewAssociation(ref)
	url := internal.URL(c, internal.AssociationPath).WithID(id).WithAction("attach")
	return c.Do(ctx, url.Request(http.MethodPost, spec), nil)
}

// DetachTag detaches a tag ID from a managed object.
// If the tag is already removed from the object, then this operation is a no-op and an error will not be thrown.
func (c *Manager) DetachTag(ctx context.Context, tagID string, ref mo.Reference) error {
	id, err := c.tagID(ctx, tagID)
	if err != nil {
		return err
	}
	spec := internal.NewAssociation(ref)
	url := internal.URL(c, internal.AssociationPath).WithID(id).WithAction("detach")
	return c.Do(ctx, url.Request(http.MethodPost, spec), nil)
}

// ListAttachedTags fetches the array of tag IDs attached to the given object.
func (c *Manager) ListAttachedTags(ctx context.Context, ref mo.Reference) ([]string, error) {
	spec := internal.NewAssociation(ref)
	url := internal.URL(c, internal.AssociationPath).WithAction("list-attached-tags")
	var res []string
	return res, c.Do(ctx, url.Request(http.MethodPost, spec), &res)
}

// GetAttachedTags fetches the array of tags attached to the given object.
func (c *Manager) GetAttachedTags(ctx context.Context, ref mo.Reference) ([]Tag, error) {
	ids, err := c.ListAttachedTags(ctx, ref)
	if err != nil {
		return nil, fmt.Errorf("get attached tags %s: %s", ref, err)
	}

	var info []Tag
	for _, id := range ids {
		tag, err := c.GetTag(ctx, id)
		if err != nil {
			return nil, fmt.Errorf("get tag %s: %s", id, err)
		}
		info = append(info, *tag)
	}
	return info, nil
}

// ListAttachedObjects fetches the array of attached objects for the given tag ID.
func (c *Manager) ListAttachedObjects(ctx context.Context, tagID string) ([]mo.Reference, error) {
	id, err := c.tagID(ctx, tagID)
	if err != nil {
		return nil, err
	}
	url := internal.URL(c, internal.AssociationPath).WithID(id).WithAction("list-attached-objects")
	var res []internal.AssociatedObject
	if err := c.Do(ctx, url.Request(http.MethodPost, nil), &res); err != nil {
		return nil, err
	}

	refs := make([]mo.Reference, len(res))
	for i := range res {
		refs[i] = res[i]
	}
	return refs, nil
}
