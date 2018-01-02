package client

const (
	ADD_REMOVE_SERVICE_LINK_INPUT_TYPE = "addRemoveServiceLinkInput"
)

type AddRemoveServiceLinkInput struct {
	Resource

	ServiceLink ServiceLink `json:"serviceLink,omitempty" yaml:"service_link,omitempty"`
}

type AddRemoveServiceLinkInputCollection struct {
	Collection
	Data []AddRemoveServiceLinkInput `json:"data,omitempty"`
}

type AddRemoveServiceLinkInputClient struct {
	rancherClient *RancherClient
}

type AddRemoveServiceLinkInputOperations interface {
	List(opts *ListOpts) (*AddRemoveServiceLinkInputCollection, error)
	Create(opts *AddRemoveServiceLinkInput) (*AddRemoveServiceLinkInput, error)
	Update(existing *AddRemoveServiceLinkInput, updates interface{}) (*AddRemoveServiceLinkInput, error)
	ById(id string) (*AddRemoveServiceLinkInput, error)
	Delete(container *AddRemoveServiceLinkInput) error
}

func newAddRemoveServiceLinkInputClient(rancherClient *RancherClient) *AddRemoveServiceLinkInputClient {
	return &AddRemoveServiceLinkInputClient{
		rancherClient: rancherClient,
	}
}

func (c *AddRemoveServiceLinkInputClient) Create(container *AddRemoveServiceLinkInput) (*AddRemoveServiceLinkInput, error) {
	resp := &AddRemoveServiceLinkInput{}
	err := c.rancherClient.doCreate(ADD_REMOVE_SERVICE_LINK_INPUT_TYPE, container, resp)
	return resp, err
}

func (c *AddRemoveServiceLinkInputClient) Update(existing *AddRemoveServiceLinkInput, updates interface{}) (*AddRemoveServiceLinkInput, error) {
	resp := &AddRemoveServiceLinkInput{}
	err := c.rancherClient.doUpdate(ADD_REMOVE_SERVICE_LINK_INPUT_TYPE, &existing.Resource, updates, resp)
	return resp, err
}

func (c *AddRemoveServiceLinkInputClient) List(opts *ListOpts) (*AddRemoveServiceLinkInputCollection, error) {
	resp := &AddRemoveServiceLinkInputCollection{}
	err := c.rancherClient.doList(ADD_REMOVE_SERVICE_LINK_INPUT_TYPE, opts, resp)
	return resp, err
}

func (c *AddRemoveServiceLinkInputClient) ById(id string) (*AddRemoveServiceLinkInput, error) {
	resp := &AddRemoveServiceLinkInput{}
	err := c.rancherClient.doById(ADD_REMOVE_SERVICE_LINK_INPUT_TYPE, id, resp)
	if apiError, ok := err.(*ApiError); ok {
		if apiError.StatusCode == 404 {
			return nil, nil
		}
	}
	return resp, err
}

func (c *AddRemoveServiceLinkInputClient) Delete(container *AddRemoveServiceLinkInput) error {
	return c.rancherClient.doResourceDelete(ADD_REMOVE_SERVICE_LINK_INPUT_TYPE, &container.Resource)
}
