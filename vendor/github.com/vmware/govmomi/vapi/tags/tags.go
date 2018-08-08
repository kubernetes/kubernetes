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
	"io"
	"net/http"
	"strings"
)

const (
	TagURL = "/com/vmware/cis/tagging/tag"
)

type TagCreateSpec struct {
	CreateSpec TagCreate `json:"create_spec"`
}

type TagCreate struct {
	CategoryID  string `json:"category_id"`
	Description string `json:"description"`
	Name        string `json:"name"`
}

type TagUpdateSpec struct {
	UpdateSpec TagUpdate `json:"update_spec,omitempty"`
}

type TagUpdate struct {
	Description string `json:"description,omitempty"`
	Name        string `json:"name,omitempty"`
}

type Tag struct {
	ID          string   `json:"id"`
	Description string   `json:"description"`
	Name        string   `json:"name"`
	CategoryID  string   `json:"category_id"`
	UsedBy      []string `json:"used_by"`
}

func (c *RestClient) CreateTagIfNotExist(ctx context.Context, name string, description string, categoryID string) (*string, error) {
	tagCreate := TagCreate{categoryID, description, name}
	spec := TagCreateSpec{tagCreate}
	id, err := c.CreateTag(ctx, &spec)
	if err == nil {
		return id, nil
	}
	// if already exists, query back
	if strings.Contains(err.Error(), ErrAlreadyExists) {
		tagObjs, err := c.GetTagByNameForCategory(ctx, name, categoryID)
		if err != nil {
			return nil, err
		}
		if tagObjs != nil {
			return &tagObjs[0].ID, nil
		}

		// should not happen
		return nil, fmt.Errorf("failed to create tag for it's existed, but could not query back. Please check system")
	}

	return nil, fmt.Errorf("created tag failed for %s", err)
}

func (c *RestClient) DeleteTagIfNoObjectAttached(ctx context.Context, id string) error {
	objs, err := c.ListAttachedObjects(ctx, id)
	if err != nil {
		return err
	}
	if len(objs) > 0 {
		return fmt.Errorf("tag %s related objects is not empty, do not delete it", id)
	}
	return c.DeleteTag(ctx, id)
}

func (c *RestClient) CreateTag(ctx context.Context, spec *TagCreateSpec) (*string, error) {
	stream, _, status, err := c.call(ctx, http.MethodPost, TagURL, spec, nil)

	if status != http.StatusOK || err != nil {
		return nil, fmt.Errorf("create tag failed with status code: %d, error message: %s", status, err)
	}

	type RespValue struct {
		Value string
	}

	var pID RespValue
	if err := json.NewDecoder(stream).Decode(&pID); err != nil {
		return nil, fmt.Errorf("decode response body failed for: %s", err)
	}
	return &pID.Value, nil
}

func (c *RestClient) GetTag(ctx context.Context, id string) (*Tag, error) {
	stream, _, status, err := c.call(ctx, http.MethodGet, fmt.Sprintf("%s/id:%s", TagURL, id), nil, nil)

	if status != http.StatusOK || err != nil {
		return nil, fmt.Errorf("get tag failed with status code: %d, error message: %s", status, err)
	}

	type RespValue struct {
		Value Tag
	}

	var pTag RespValue
	if err := json.NewDecoder(stream).Decode(&pTag); err != nil {
		return nil, fmt.Errorf("decode response body failed for: %s", err)
	}
	return &(pTag.Value), nil
}

func (c *RestClient) UpdateTag(ctx context.Context, id string, spec *TagUpdateSpec) error {
	_, _, status, err := c.call(ctx, http.MethodPatch, fmt.Sprintf("%s/id:%s", TagURL, id), spec, nil)

	if status != http.StatusOK || err != nil {
		return fmt.Errorf("update tag failed with status code: %d, error message: %s", status, err)
	}

	return nil
}

func (c *RestClient) DeleteTag(ctx context.Context, id string) error {
	_, _, status, err := c.call(ctx, http.MethodDelete, fmt.Sprintf("%s/id:%s", TagURL, id), nil, nil)

	if status != http.StatusOK || err != nil {
		return fmt.Errorf("delete tag failed with status code: %d, error message: %s", status, err)
	}
	return nil
}

func (c *RestClient) ListTags(ctx context.Context) ([]string, error) {
	stream, _, status, err := c.call(ctx, http.MethodGet, TagURL, nil, nil)

	if status != http.StatusOK || err != nil {
		return nil, fmt.Errorf("get tags failed with status code: %d, error message: %s", status, err)
	}

	return c.handleTagIDList(stream)
}

type TagsInfo struct {
	Name  string
	TagID string
}

func (c *RestClient) ListTagsByName(ctx context.Context) ([]TagsInfo, error) {
	tagIds, err := c.ListTags(ctx)
	if err != nil {
		return nil, fmt.Errorf("get tags failed for: %s", err)
	}

	var tagsInfoSlice []TagsInfo
	for _, cID := range tagIds {
		tag, err := c.GetTag(ctx, cID)
		if err != nil {
			return nil, fmt.Errorf("get category %s failed for %s", cID, err)
		}
		tagsCreate := &TagsInfo{Name: tag.Name, TagID: tag.ID}

		tagsInfoSlice = append(tagsInfoSlice, *tagsCreate)

	}
	return tagsInfoSlice, nil
}

func (c *RestClient) ListTagsForCategory(ctx context.Context, id string) ([]string, error) {

	type PostCategory struct {
		ID string `json:"category_id"`
	}
	spec := PostCategory{id}
	stream, _, status, err := c.call(ctx, http.MethodPost, fmt.Sprintf("%s/id:%s?~action=list-tags-for-category", TagURL, id), spec, nil)

	if status != http.StatusOK || err != nil {
		return nil, fmt.Errorf("list tags for category failed with status code: %d, error message: %s", status, err)
	}

	return c.handleTagIDList(stream)
}

func (c *RestClient) ListTagsInfoForCategory(ctx context.Context, id string) ([]TagsInfo, error) {

	type PostCategory struct {
		ID string `json:"category_id"`
	}
	spec := PostCategory{id}
	stream, _, status, err := c.call(ctx, http.MethodPost, fmt.Sprintf("%s/id:%s?~action=list-tags-for-category", TagURL, id), spec, nil)

	if status != http.StatusOK || err != nil {
		return nil, fmt.Errorf("list tags for category failed with status code: %d, error message: %s", status, err)
	}
	var tagsInfoSlice []TagsInfo
	tmp, err := c.handleTagIDList(stream)
	for _, item := range tmp {
		tag, err := c.GetTag(ctx, item)
		if err != nil {
			return nil, fmt.Errorf("get category %s failed for %s", item, err)
		}
		tagsCreate := &TagsInfo{Name: tag.Name, TagID: tag.ID}

		tagsInfoSlice = append(tagsInfoSlice, *tagsCreate)
	}
	return tagsInfoSlice, nil
}

func (c *RestClient) handleTagIDList(stream io.ReadCloser) ([]string, error) {
	type Tags struct {
		Value []string
	}

	var pTags Tags
	if err := json.NewDecoder(stream).Decode(&pTags); err != nil {
		return nil, fmt.Errorf("decode response body failed for: %s", err)
	}
	return pTags.Value, nil
}

// Get tag through tag name and category id
func (c *RestClient) GetTagByNameForCategory(ctx context.Context, name string, id string) ([]Tag, error) {
	tagIds, err := c.ListTagsForCategory(ctx, id)
	if err != nil {
		return nil, fmt.Errorf("get tag failed for %s", err)
	}

	var tags []Tag
	for _, tID := range tagIds {
		tag, err := c.GetTag(ctx, tID)
		if err != nil {
			return nil, fmt.Errorf("get tag %s failed for %s", tID, err)
		}
		if tag.Name == name {
			tags = append(tags, *tag)
		}
	}
	return tags, nil
}
