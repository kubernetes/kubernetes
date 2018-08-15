// Copyright 2017 VMware, Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package tags

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"

	"github.com/vmware/govmomi/vim25/types"
)

const (
	TagAssociationURL = "/com/vmware/cis/tagging/tag-association"
)

type AssociatedObject struct {
	ID   string `json:"id"`
	Type string `json:"type"`
}

type TagAssociationSpec struct {
	ObjectID *AssociatedObject `json:"object_id,omitempty"`
	TagID    *string           `json:"tag_id,omitempty"`
}

type AttachedTagsInfo struct {
	Name  string
	TagID string
}

func (c *RestClient) getAssociatedObject(ref *types.ManagedObjectReference) *AssociatedObject {
	if ref == nil {
		return nil
	}
	object := AssociatedObject{
		ID:   ref.Value,
		Type: ref.Type,
	}
	return &object
}

func (c *RestClient) getAssociationSpec(tagID *string, ref *types.ManagedObjectReference) *TagAssociationSpec {
	object := c.getAssociatedObject(ref)
	spec := TagAssociationSpec{
		TagID:    tagID,
		ObjectID: object,
	}
	return &spec
}

func (c *RestClient) AttachTagToObject(ctx context.Context, tagID string, ref *types.ManagedObjectReference) error {
	spec := c.getAssociationSpec(&tagID, ref)
	_, _, status, err := c.call(ctx, http.MethodPost, fmt.Sprintf("%s?~action=attach", TagAssociationURL), *spec, nil)

	if status != http.StatusOK || err != nil {
		return fmt.Errorf("attach tag failed with status code: %d, error message: %s", status, err)
	}
	return nil
}

func (c *RestClient) DetachTagFromObject(ctx context.Context, tagID string, ref *types.ManagedObjectReference) error {
	spec := c.getAssociationSpec(&tagID, ref)
	_, _, status, err := c.call(ctx, http.MethodPost, fmt.Sprintf("%s?~action=detach", TagAssociationURL), *spec, nil)

	if status != http.StatusOK || err != nil {
		return fmt.Errorf("detach tag failed with status code: %d, error message: %s", status, err)
	}
	return nil
}

func (c *RestClient) ListAttachedTags(ctx context.Context, ref *types.ManagedObjectReference) ([]string, error) {
	spec := c.getAssociationSpec(nil, ref)
	stream, _, status, err := c.call(ctx, http.MethodPost, fmt.Sprintf("%s?~action=list-attached-tags", TagAssociationURL), *spec, nil)

	if status != http.StatusOK || err != nil {
		return nil, fmt.Errorf("detach tag failed with status code: %d, error message: %s", status, err)
	}

	type RespValue struct {
		Value []string
	}

	var pTag RespValue
	if err := json.NewDecoder(stream).Decode(&pTag); err != nil {
		return nil, fmt.Errorf("decode response body failed for: %s", err)
	}
	return pTag.Value, nil
}

func (c *RestClient) ListAttachedTagsByName(ctx context.Context, ref *types.ManagedObjectReference) ([]AttachedTagsInfo, error) {
	tagIds, err := c.ListAttachedTags(ctx, ref)
	if err != nil {
		return nil, fmt.Errorf("get attached tag failed for: %s", err)
	}

	var attachedTagsInfoSlice []AttachedTagsInfo
	for _, cID := range tagIds {
		tag, err := c.GetTag(ctx, cID)
		if err != nil {
			return nil, fmt.Errorf("get tag %s failed for %s", cID, err)
		}
		attachedTagsCreate := &AttachedTagsInfo{Name: tag.Name, TagID: tag.ID}
		attachedTagsInfoSlice = append(attachedTagsInfoSlice, *attachedTagsCreate)
	}
	return attachedTagsInfoSlice, nil
}

func (c *RestClient) ListAttachedObjects(ctx context.Context, tagID string) ([]AssociatedObject, error) {
	spec := c.getAssociationSpec(&tagID, nil)
	stream, _, status, err := c.call(ctx, http.MethodPost, fmt.Sprintf("%s?~action=list-attached-objects", TagAssociationURL), *spec, nil)
	if status != http.StatusOK || err != nil {
		return nil, fmt.Errorf("list object failed with status code: %d, error message: %s", status, err)
	}

	type RespValue struct {
		Value []AssociatedObject
	}

	var pTag RespValue
	if err := json.NewDecoder(stream).Decode(&pTag); err != nil {
		return nil, fmt.Errorf("decode response body failed for: %s", err)
	}

	return pTag.Value, nil
}
