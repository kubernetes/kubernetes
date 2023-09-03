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
	"encoding/json"
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
	url := c.Resource(internal.AssociationPath).WithID(id).WithAction("attach")
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
	url := c.Resource(internal.AssociationPath).WithID(id).WithAction("detach")
	return c.Do(ctx, url.Request(http.MethodPost, spec), nil)
}

// batchResponse is the response type used by attach/detach operations which
// take multiple tagIDs or moRefs as input. On failure Success will be false and
// Errors contains information about all failed operations
type batchResponse struct {
	Success bool        `json:"success"`
	Errors  BatchErrors `json:"error_messages,omitempty"`
}

// AttachTagToMultipleObjects attaches a tag ID to multiple managed objects.
// This operation is idempotent, i.e. if a tag is already attached to the
// object, then the individual operation is a no-op and no error will be thrown.
//
// This operation was added in vSphere API 6.5.
func (c *Manager) AttachTagToMultipleObjects(ctx context.Context, tagID string, refs []mo.Reference) error {
	id, err := c.tagID(ctx, tagID)
	if err != nil {
		return err
	}

	var ids []internal.AssociatedObject
	for i := range refs {
		ids = append(ids, internal.AssociatedObject(refs[i].Reference()))
	}

	spec := struct {
		ObjectIDs []internal.AssociatedObject `json:"object_ids"`
	}{ids}

	url := c.Resource(internal.AssociationPath).WithID(id).WithAction("attach-tag-to-multiple-objects")
	return c.Do(ctx, url.Request(http.MethodPost, spec), nil)
}

// AttachMultipleTagsToObject attaches multiple tag IDs to a managed object.
// This operation is idempotent. If a tag is already attached to the object,
// then the individual operation is a no-op and no error will be thrown. This
// operation is not atomic. If the underlying call fails with one or more tags
// not successfully attached to the managed object reference it might leave the
// managed object reference in a partially tagged state and needs to be resolved
// by the caller. In this case BatchErrors is returned and can be used to
// analyse failure reasons on each failed tag.
//
// Specified tagIDs must use URN-notation instead of display names or a generic
// error will be returned and no tagging operation will be performed. If the
// managed object reference does not exist a generic 403 Forbidden error will be
// returned.
//
// This operation was added in vSphere API 6.5.
func (c *Manager) AttachMultipleTagsToObject(ctx context.Context, tagIDs []string, ref mo.Reference) error {
	for _, id := range tagIDs {
		// URN enforced to avoid unnecessary round-trips due to invalid tags or display
		// name lookups
		if isName(id) {
			return fmt.Errorf("specified tag is not a URN: %q", id)
		}
	}

	obj := internal.AssociatedObject(ref.Reference())
	spec := struct {
		ObjectID internal.AssociatedObject `json:"object_id"`
		TagIDs   []string                  `json:"tag_ids"`
	}{
		ObjectID: obj,
		TagIDs:   tagIDs,
	}

	var res batchResponse
	url := c.Resource(internal.AssociationPath).WithAction("attach-multiple-tags-to-object")
	err := c.Do(ctx, url.Request(http.MethodPost, spec), &res)
	if err != nil {
		return err
	}

	if !res.Success {
		if len(res.Errors) != 0 {
			return res.Errors
		}
		panic("invalid batch error")
	}

	return nil
}

// DetachMultipleTagsFromObject detaches multiple tag IDs from a managed object.
// This operation is idempotent. If a tag is already detached from the object,
// then the individual operation is a no-op and no error will be thrown. This
// operation is not atomic. If the underlying call fails with one or more tags
// not successfully detached from the managed object reference it might leave
// the managed object reference in a partially tagged state and needs to be
// resolved by the caller. In this case BatchErrors is returned and can be used
// to analyse failure reasons on each failed tag.
//
// Specified tagIDs must use URN-notation instead of display names or a generic
// error will be returned and no tagging operation will be performed. If the
// managed object reference does not exist a generic 403 Forbidden error will be
// returned.
//
// This operation was added in vSphere API 6.5.
func (c *Manager) DetachMultipleTagsFromObject(ctx context.Context, tagIDs []string, ref mo.Reference) error {
	for _, id := range tagIDs {
		// URN enforced to avoid unnecessary round-trips due to invalid tags or display
		// name lookups
		if isName(id) {
			return fmt.Errorf("specified tag is not a URN: %q", id)
		}
	}

	obj := internal.AssociatedObject(ref.Reference())
	spec := struct {
		ObjectID internal.AssociatedObject `json:"object_id"`
		TagIDs   []string                  `json:"tag_ids"`
	}{
		ObjectID: obj,
		TagIDs:   tagIDs,
	}

	var res batchResponse
	url := c.Resource(internal.AssociationPath).WithAction("detach-multiple-tags-from-object")
	err := c.Do(ctx, url.Request(http.MethodPost, spec), &res)
	if err != nil {
		return err
	}

	if !res.Success {
		if len(res.Errors) != 0 {
			return res.Errors
		}
		panic("invalid batch error")
	}

	return nil
}

// ListAttachedTags fetches the array of tag IDs attached to the given object.
func (c *Manager) ListAttachedTags(ctx context.Context, ref mo.Reference) ([]string, error) {
	spec := internal.NewAssociation(ref)
	url := c.Resource(internal.AssociationPath).WithAction("list-attached-tags")
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
	url := c.Resource(internal.AssociationPath).WithID(id).WithAction("list-attached-objects")
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

// AttachedObjects is the response type used by ListAttachedObjectsOnTags.
type AttachedObjects struct {
	TagID     string         `json:"tag_id"`
	Tag       *Tag           `json:"tag,omitempty"`
	ObjectIDs []mo.Reference `json:"object_ids"`
}

// UnmarshalJSON implements json.Unmarshaler.
func (t *AttachedObjects) UnmarshalJSON(b []byte) error {
	var o struct {
		TagID     string                      `json:"tag_id"`
		ObjectIDs []internal.AssociatedObject `json:"object_ids"`
	}
	err := json.Unmarshal(b, &o)
	if err != nil {
		return err
	}

	t.TagID = o.TagID
	t.ObjectIDs = make([]mo.Reference, len(o.ObjectIDs))
	for i := range o.ObjectIDs {
		t.ObjectIDs[i] = o.ObjectIDs[i]
	}

	return nil
}

// ListAttachedObjectsOnTags fetches the array of attached objects for the given tag IDs.
func (c *Manager) ListAttachedObjectsOnTags(ctx context.Context, tagID []string) ([]AttachedObjects, error) {
	var ids []string
	for i := range tagID {
		id, err := c.tagID(ctx, tagID[i])
		if err != nil {
			return nil, err
		}
		ids = append(ids, id)
	}

	spec := struct {
		TagIDs []string `json:"tag_ids"`
	}{ids}

	url := c.Resource(internal.AssociationPath).WithAction("list-attached-objects-on-tags")
	var res []AttachedObjects
	return res, c.Do(ctx, url.Request(http.MethodPost, spec), &res)
}

// GetAttachedObjectsOnTags combines ListAttachedObjectsOnTags and populates each Tag field.
func (c *Manager) GetAttachedObjectsOnTags(ctx context.Context, tagID []string) ([]AttachedObjects, error) {
	objs, err := c.ListAttachedObjectsOnTags(ctx, tagID)
	if err != nil {
		return nil, fmt.Errorf("list attached objects %s: %s", tagID, err)
	}

	tags := make(map[string]*Tag)

	for i := range objs {
		var err error
		id := objs[i].TagID
		tag, ok := tags[id]
		if !ok {
			tag, err = c.GetTag(ctx, id)
			if err != nil {
				return nil, fmt.Errorf("get tag %s: %s", id, err)
			}
			objs[i].Tag = tag
		}
	}

	return objs, nil
}

// AttachedTags is the response type used by ListAttachedTagsOnObjects.
type AttachedTags struct {
	ObjectID mo.Reference `json:"object_id"`
	TagIDs   []string     `json:"tag_ids"`
	Tags     []Tag        `json:"tags,omitempty"`
}

// UnmarshalJSON implements json.Unmarshaler.
func (t *AttachedTags) UnmarshalJSON(b []byte) error {
	var o struct {
		ObjectID internal.AssociatedObject `json:"object_id"`
		TagIDs   []string                  `json:"tag_ids"`
	}
	err := json.Unmarshal(b, &o)
	if err != nil {
		return err
	}

	t.ObjectID = o.ObjectID
	t.TagIDs = o.TagIDs

	return nil
}

// ListAttachedTagsOnObjects fetches the array of attached tag IDs for the given object IDs.
func (c *Manager) ListAttachedTagsOnObjects(ctx context.Context, objectID []mo.Reference) ([]AttachedTags, error) {
	var ids []internal.AssociatedObject
	for i := range objectID {
		ids = append(ids, internal.AssociatedObject(objectID[i].Reference()))
	}

	spec := struct {
		ObjectIDs []internal.AssociatedObject `json:"object_ids"`
	}{ids}

	url := c.Resource(internal.AssociationPath).WithAction("list-attached-tags-on-objects")
	var res []AttachedTags
	return res, c.Do(ctx, url.Request(http.MethodPost, spec), &res)
}

// GetAttachedTagsOnObjects calls ListAttachedTagsOnObjects and populates each Tags field.
func (c *Manager) GetAttachedTagsOnObjects(ctx context.Context, objectID []mo.Reference) ([]AttachedTags, error) {
	objs, err := c.ListAttachedTagsOnObjects(ctx, objectID)
	if err != nil {
		return nil, fmt.Errorf("list attached tags %s: %s", objectID, err)
	}

	tags := make(map[string]*Tag)

	for i := range objs {
		for _, id := range objs[i].TagIDs {
			var err error
			tag, ok := tags[id]
			if !ok {
				tag, err = c.GetTag(ctx, id)
				if err != nil {
					return nil, fmt.Errorf("get tag %s: %s", id, err)
				}
				tags[id] = tag
			}
			objs[i].Tags = append(objs[i].Tags, *tag)
		}
	}

	return objs, nil
}
