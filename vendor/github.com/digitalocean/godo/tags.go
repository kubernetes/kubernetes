package godo

import "fmt"

const tagsBasePath = "v2/tags"

// TagsService is an interface for interfacing with the tags
// endpoints of the DigitalOcean API
// See: https://developers.digitalocean.com/documentation/v2#tags
type TagsService interface {
	List(*ListOptions) ([]Tag, *Response, error)
	Get(string) (*Tag, *Response, error)
	Create(*TagCreateRequest) (*Tag, *Response, error)
	Update(string, *TagUpdateRequest) (*Response, error)
	Delete(string) (*Response, error)

	TagResources(string, *TagResourcesRequest) (*Response, error)
	UntagResources(string, *UntagResourcesRequest) (*Response, error)
}

// TagsServiceOp handles communication with tag related method of the
// DigitalOcean API.
type TagsServiceOp struct {
	client *Client
}

var _ TagsService = &TagsServiceOp{}

// ResourceType represents a class of resource, currently only droplet are supported
type ResourceType string

const (
	DropletResourceType ResourceType = "droplet"
)

// Resource represent a single resource for associating/disassociating with tags
type Resource struct {
	ID   string       `json:"resource_id,omit_empty"`
	Type ResourceType `json:"resource_type,omit_empty"`
}

// TaggedResources represent the set of resources a tag is attached to
type TaggedResources struct {
	Droplets *TaggedDropletsResources `json:"droplets,omitempty"`
}

// TaggedDropletsResources represent the droplet resources a tag is attached to
type TaggedDropletsResources struct {
	Count      int      `json:"count,float64,omitempty"`
	LastTagged *Droplet `json:"last_tagged,omitempty"`
}

// Tag represent DigitalOcean tag
type Tag struct {
	Name      string           `json:"name,omitempty"`
	Resources *TaggedResources `json:"resources,omitempty"`
}

type TagCreateRequest struct {
	Name string `json:"name"`
}

type TagUpdateRequest struct {
	Name string `json:"name"`
}

type TagResourcesRequest struct {
	Resources []Resource `json:"resources"`
}

type UntagResourcesRequest struct {
	Resources []Resource `json:"resources"`
}

type tagsRoot struct {
	Tags  []Tag  `json:"tags"`
	Links *Links `json:"links"`
}

type tagRoot struct {
	Tag *Tag `json:"tag"`
}

// List all tags
func (s *TagsServiceOp) List(opt *ListOptions) ([]Tag, *Response, error) {
	path := tagsBasePath
	path, err := addOptions(path, opt)

	if err != nil {
		return nil, nil, err
	}

	req, err := s.client.NewRequest("GET", path, nil)
	if err != nil {
		return nil, nil, err
	}

	root := new(tagsRoot)
	resp, err := s.client.Do(req, root)
	if err != nil {
		return nil, resp, err
	}
	if l := root.Links; l != nil {
		resp.Links = l
	}

	return root.Tags, resp, err
}

// Get a single tag
func (s *TagsServiceOp) Get(name string) (*Tag, *Response, error) {
	path := fmt.Sprintf("%s/%s", tagsBasePath, name)

	req, err := s.client.NewRequest("GET", path, nil)
	if err != nil {
		return nil, nil, err
	}

	root := new(tagRoot)
	resp, err := s.client.Do(req, root)
	if err != nil {
		return nil, resp, err
	}

	return root.Tag, resp, err
}

// Create a new tag
func (s *TagsServiceOp) Create(createRequest *TagCreateRequest) (*Tag, *Response, error) {
	if createRequest == nil {
		return nil, nil, NewArgError("createRequest", "cannot be nil")
	}

	req, err := s.client.NewRequest("POST", tagsBasePath, createRequest)
	if err != nil {
		return nil, nil, err
	}

	root := new(tagRoot)
	resp, err := s.client.Do(req, root)
	if err != nil {
		return nil, resp, err
	}

	return root.Tag, resp, err
}

// Update an exsting tag
func (s *TagsServiceOp) Update(name string, updateRequest *TagUpdateRequest) (*Response, error) {
	if name == "" {
		return nil, NewArgError("name", "cannot be empty")
	}

	if updateRequest == nil {
		return nil, NewArgError("updateRequest", "cannot be nil")
	}

	path := fmt.Sprintf("%s/%s", tagsBasePath, name)
	req, err := s.client.NewRequest("PUT", path, updateRequest)
	if err != nil {
		return nil, err
	}

	resp, err := s.client.Do(req, nil)

	return resp, err
}

// Delete an existing tag
func (s *TagsServiceOp) Delete(name string) (*Response, error) {
	if name == "" {
		return nil, NewArgError("name", "cannot be empty")
	}

	path := fmt.Sprintf("%s/%s", tagsBasePath, name)
	req, err := s.client.NewRequest("DELETE", path, nil)
	if err != nil {
		return nil, err
	}

	resp, err := s.client.Do(req, nil)

	return resp, err
}

// Associate resources with a tag
func (s *TagsServiceOp) TagResources(name string, tagRequest *TagResourcesRequest) (*Response, error) {
	if name == "" {
		return nil, NewArgError("name", "cannot be empty")
	}

	if tagRequest == nil {
		return nil, NewArgError("tagRequest", "cannot be nil")
	}

	path := fmt.Sprintf("%s/%s/resources", tagsBasePath, name)
	req, err := s.client.NewRequest("POST", path, tagRequest)
	if err != nil {
		return nil, err
	}

	resp, err := s.client.Do(req, nil)

	return resp, err
}

// Dissociate resources with a tag
func (s *TagsServiceOp) UntagResources(name string, untagRequest *UntagResourcesRequest) (*Response, error) {
	if name == "" {
		return nil, NewArgError("name", "cannot be empty")
	}

	if untagRequest == nil {
		return nil, NewArgError("tagRequest", "cannot be nil")
	}

	path := fmt.Sprintf("%s/%s/resources", tagsBasePath, name)
	req, err := s.client.NewRequest("DELETE", path, untagRequest)
	if err != nil {
		return nil, err
	}

	resp, err := s.client.Do(req, nil)

	return resp, err
}
