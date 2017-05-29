package client

const (
	LOAD_BALANCER_SERVICE_LINK_TYPE = "loadBalancerServiceLink"
)

type LoadBalancerServiceLink struct {
	Resource

	Ports []string `json:"ports,omitempty" yaml:"ports,omitempty"`

	ServiceId string `json:"serviceId,omitempty" yaml:"service_id,omitempty"`

	Uuid string `json:"uuid,omitempty" yaml:"uuid,omitempty"`
}

type LoadBalancerServiceLinkCollection struct {
	Collection
	Data []LoadBalancerServiceLink `json:"data,omitempty"`
}

type LoadBalancerServiceLinkClient struct {
	rancherClient *RancherClient
}

type LoadBalancerServiceLinkOperations interface {
	List(opts *ListOpts) (*LoadBalancerServiceLinkCollection, error)
	Create(opts *LoadBalancerServiceLink) (*LoadBalancerServiceLink, error)
	Update(existing *LoadBalancerServiceLink, updates interface{}) (*LoadBalancerServiceLink, error)
	ById(id string) (*LoadBalancerServiceLink, error)
	Delete(container *LoadBalancerServiceLink) error
}

func newLoadBalancerServiceLinkClient(rancherClient *RancherClient) *LoadBalancerServiceLinkClient {
	return &LoadBalancerServiceLinkClient{
		rancherClient: rancherClient,
	}
}

func (c *LoadBalancerServiceLinkClient) Create(container *LoadBalancerServiceLink) (*LoadBalancerServiceLink, error) {
	resp := &LoadBalancerServiceLink{}
	err := c.rancherClient.doCreate(LOAD_BALANCER_SERVICE_LINK_TYPE, container, resp)
	return resp, err
}

func (c *LoadBalancerServiceLinkClient) Update(existing *LoadBalancerServiceLink, updates interface{}) (*LoadBalancerServiceLink, error) {
	resp := &LoadBalancerServiceLink{}
	err := c.rancherClient.doUpdate(LOAD_BALANCER_SERVICE_LINK_TYPE, &existing.Resource, updates, resp)
	return resp, err
}

func (c *LoadBalancerServiceLinkClient) List(opts *ListOpts) (*LoadBalancerServiceLinkCollection, error) {
	resp := &LoadBalancerServiceLinkCollection{}
	err := c.rancherClient.doList(LOAD_BALANCER_SERVICE_LINK_TYPE, opts, resp)
	return resp, err
}

func (c *LoadBalancerServiceLinkClient) ById(id string) (*LoadBalancerServiceLink, error) {
	resp := &LoadBalancerServiceLink{}
	err := c.rancherClient.doById(LOAD_BALANCER_SERVICE_LINK_TYPE, id, resp)
	if apiError, ok := err.(*ApiError); ok {
		if apiError.StatusCode == 404 {
			return nil, nil
		}
	}
	return resp, err
}

func (c *LoadBalancerServiceLinkClient) Delete(container *LoadBalancerServiceLink) error {
	return c.rancherClient.doResourceDelete(LOAD_BALANCER_SERVICE_LINK_TYPE, &container.Resource)
}
