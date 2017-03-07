package client

const (
	PUBLIC_ENDPOINT_TYPE = "publicEndpoint"
)

type PublicEndpoint struct {
	Resource

	HostId string `json:"hostId,omitempty" yaml:"host_id,omitempty"`

	InstanceId string `json:"instanceId,omitempty" yaml:"instance_id,omitempty"`

	IpAddress string `json:"ipAddress,omitempty" yaml:"ip_address,omitempty"`

	Port int64 `json:"port,omitempty" yaml:"port,omitempty"`

	ServiceId string `json:"serviceId,omitempty" yaml:"service_id,omitempty"`
}

type PublicEndpointCollection struct {
	Collection
	Data []PublicEndpoint `json:"data,omitempty"`
}

type PublicEndpointClient struct {
	rancherClient *RancherClient
}

type PublicEndpointOperations interface {
	List(opts *ListOpts) (*PublicEndpointCollection, error)
	Create(opts *PublicEndpoint) (*PublicEndpoint, error)
	Update(existing *PublicEndpoint, updates interface{}) (*PublicEndpoint, error)
	ById(id string) (*PublicEndpoint, error)
	Delete(container *PublicEndpoint) error
}

func newPublicEndpointClient(rancherClient *RancherClient) *PublicEndpointClient {
	return &PublicEndpointClient{
		rancherClient: rancherClient,
	}
}

func (c *PublicEndpointClient) Create(container *PublicEndpoint) (*PublicEndpoint, error) {
	resp := &PublicEndpoint{}
	err := c.rancherClient.doCreate(PUBLIC_ENDPOINT_TYPE, container, resp)
	return resp, err
}

func (c *PublicEndpointClient) Update(existing *PublicEndpoint, updates interface{}) (*PublicEndpoint, error) {
	resp := &PublicEndpoint{}
	err := c.rancherClient.doUpdate(PUBLIC_ENDPOINT_TYPE, &existing.Resource, updates, resp)
	return resp, err
}

func (c *PublicEndpointClient) List(opts *ListOpts) (*PublicEndpointCollection, error) {
	resp := &PublicEndpointCollection{}
	err := c.rancherClient.doList(PUBLIC_ENDPOINT_TYPE, opts, resp)
	return resp, err
}

func (c *PublicEndpointClient) ById(id string) (*PublicEndpoint, error) {
	resp := &PublicEndpoint{}
	err := c.rancherClient.doById(PUBLIC_ENDPOINT_TYPE, id, resp)
	if apiError, ok := err.(*ApiError); ok {
		if apiError.StatusCode == 404 {
			return nil, nil
		}
	}
	return resp, err
}

func (c *PublicEndpointClient) Delete(container *PublicEndpoint) error {
	return c.rancherClient.doResourceDelete(PUBLIC_ENDPOINT_TYPE, &container.Resource)
}
