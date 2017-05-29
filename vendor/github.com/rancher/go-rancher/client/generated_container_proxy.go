package client

const (
	CONTAINER_PROXY_TYPE = "containerProxy"
)

type ContainerProxy struct {
	Resource

	Port int64 `json:"port,omitempty" yaml:"port,omitempty"`

	Scheme string `json:"scheme,omitempty" yaml:"scheme,omitempty"`
}

type ContainerProxyCollection struct {
	Collection
	Data []ContainerProxy `json:"data,omitempty"`
}

type ContainerProxyClient struct {
	rancherClient *RancherClient
}

type ContainerProxyOperations interface {
	List(opts *ListOpts) (*ContainerProxyCollection, error)
	Create(opts *ContainerProxy) (*ContainerProxy, error)
	Update(existing *ContainerProxy, updates interface{}) (*ContainerProxy, error)
	ById(id string) (*ContainerProxy, error)
	Delete(container *ContainerProxy) error
}

func newContainerProxyClient(rancherClient *RancherClient) *ContainerProxyClient {
	return &ContainerProxyClient{
		rancherClient: rancherClient,
	}
}

func (c *ContainerProxyClient) Create(container *ContainerProxy) (*ContainerProxy, error) {
	resp := &ContainerProxy{}
	err := c.rancherClient.doCreate(CONTAINER_PROXY_TYPE, container, resp)
	return resp, err
}

func (c *ContainerProxyClient) Update(existing *ContainerProxy, updates interface{}) (*ContainerProxy, error) {
	resp := &ContainerProxy{}
	err := c.rancherClient.doUpdate(CONTAINER_PROXY_TYPE, &existing.Resource, updates, resp)
	return resp, err
}

func (c *ContainerProxyClient) List(opts *ListOpts) (*ContainerProxyCollection, error) {
	resp := &ContainerProxyCollection{}
	err := c.rancherClient.doList(CONTAINER_PROXY_TYPE, opts, resp)
	return resp, err
}

func (c *ContainerProxyClient) ById(id string) (*ContainerProxy, error) {
	resp := &ContainerProxy{}
	err := c.rancherClient.doById(CONTAINER_PROXY_TYPE, id, resp)
	if apiError, ok := err.(*ApiError); ok {
		if apiError.StatusCode == 404 {
			return nil, nil
		}
	}
	return resp, err
}

func (c *ContainerProxyClient) Delete(container *ContainerProxy) error {
	return c.rancherClient.doResourceDelete(CONTAINER_PROXY_TYPE, &container.Resource)
}
