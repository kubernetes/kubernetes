package client

const (
	SET_LOAD_BALANCER_SERVICE_LINKS_INPUT_TYPE = "setLoadBalancerServiceLinksInput"
)

type SetLoadBalancerServiceLinksInput struct {
	Resource

	ServiceLinks []interface{} `json:"serviceLinks,omitempty" yaml:"service_links,omitempty"`
}

type SetLoadBalancerServiceLinksInputCollection struct {
	Collection
	Data []SetLoadBalancerServiceLinksInput `json:"data,omitempty"`
}

type SetLoadBalancerServiceLinksInputClient struct {
	rancherClient *RancherClient
}

type SetLoadBalancerServiceLinksInputOperations interface {
	List(opts *ListOpts) (*SetLoadBalancerServiceLinksInputCollection, error)
	Create(opts *SetLoadBalancerServiceLinksInput) (*SetLoadBalancerServiceLinksInput, error)
	Update(existing *SetLoadBalancerServiceLinksInput, updates interface{}) (*SetLoadBalancerServiceLinksInput, error)
	ById(id string) (*SetLoadBalancerServiceLinksInput, error)
	Delete(container *SetLoadBalancerServiceLinksInput) error
}

func newSetLoadBalancerServiceLinksInputClient(rancherClient *RancherClient) *SetLoadBalancerServiceLinksInputClient {
	return &SetLoadBalancerServiceLinksInputClient{
		rancherClient: rancherClient,
	}
}

func (c *SetLoadBalancerServiceLinksInputClient) Create(container *SetLoadBalancerServiceLinksInput) (*SetLoadBalancerServiceLinksInput, error) {
	resp := &SetLoadBalancerServiceLinksInput{}
	err := c.rancherClient.doCreate(SET_LOAD_BALANCER_SERVICE_LINKS_INPUT_TYPE, container, resp)
	return resp, err
}

func (c *SetLoadBalancerServiceLinksInputClient) Update(existing *SetLoadBalancerServiceLinksInput, updates interface{}) (*SetLoadBalancerServiceLinksInput, error) {
	resp := &SetLoadBalancerServiceLinksInput{}
	err := c.rancherClient.doUpdate(SET_LOAD_BALANCER_SERVICE_LINKS_INPUT_TYPE, &existing.Resource, updates, resp)
	return resp, err
}

func (c *SetLoadBalancerServiceLinksInputClient) List(opts *ListOpts) (*SetLoadBalancerServiceLinksInputCollection, error) {
	resp := &SetLoadBalancerServiceLinksInputCollection{}
	err := c.rancherClient.doList(SET_LOAD_BALANCER_SERVICE_LINKS_INPUT_TYPE, opts, resp)
	return resp, err
}

func (c *SetLoadBalancerServiceLinksInputClient) ById(id string) (*SetLoadBalancerServiceLinksInput, error) {
	resp := &SetLoadBalancerServiceLinksInput{}
	err := c.rancherClient.doById(SET_LOAD_BALANCER_SERVICE_LINKS_INPUT_TYPE, id, resp)
	if apiError, ok := err.(*ApiError); ok {
		if apiError.StatusCode == 404 {
			return nil, nil
		}
	}
	return resp, err
}

func (c *SetLoadBalancerServiceLinksInputClient) Delete(container *SetLoadBalancerServiceLinksInput) error {
	return c.rancherClient.doResourceDelete(SET_LOAD_BALANCER_SERVICE_LINKS_INPUT_TYPE, &container.Resource)
}
