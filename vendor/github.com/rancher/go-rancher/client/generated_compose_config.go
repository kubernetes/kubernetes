package client

const (
	COMPOSE_CONFIG_TYPE = "composeConfig"
)

type ComposeConfig struct {
	Resource

	DockerComposeConfig string `json:"dockerComposeConfig,omitempty" yaml:"docker_compose_config,omitempty"`

	RancherComposeConfig string `json:"rancherComposeConfig,omitempty" yaml:"rancher_compose_config,omitempty"`
}

type ComposeConfigCollection struct {
	Collection
	Data []ComposeConfig `json:"data,omitempty"`
}

type ComposeConfigClient struct {
	rancherClient *RancherClient
}

type ComposeConfigOperations interface {
	List(opts *ListOpts) (*ComposeConfigCollection, error)
	Create(opts *ComposeConfig) (*ComposeConfig, error)
	Update(existing *ComposeConfig, updates interface{}) (*ComposeConfig, error)
	ById(id string) (*ComposeConfig, error)
	Delete(container *ComposeConfig) error
}

func newComposeConfigClient(rancherClient *RancherClient) *ComposeConfigClient {
	return &ComposeConfigClient{
		rancherClient: rancherClient,
	}
}

func (c *ComposeConfigClient) Create(container *ComposeConfig) (*ComposeConfig, error) {
	resp := &ComposeConfig{}
	err := c.rancherClient.doCreate(COMPOSE_CONFIG_TYPE, container, resp)
	return resp, err
}

func (c *ComposeConfigClient) Update(existing *ComposeConfig, updates interface{}) (*ComposeConfig, error) {
	resp := &ComposeConfig{}
	err := c.rancherClient.doUpdate(COMPOSE_CONFIG_TYPE, &existing.Resource, updates, resp)
	return resp, err
}

func (c *ComposeConfigClient) List(opts *ListOpts) (*ComposeConfigCollection, error) {
	resp := &ComposeConfigCollection{}
	err := c.rancherClient.doList(COMPOSE_CONFIG_TYPE, opts, resp)
	return resp, err
}

func (c *ComposeConfigClient) ById(id string) (*ComposeConfig, error) {
	resp := &ComposeConfig{}
	err := c.rancherClient.doById(COMPOSE_CONFIG_TYPE, id, resp)
	if apiError, ok := err.(*ApiError); ok {
		if apiError.StatusCode == 404 {
			return nil, nil
		}
	}
	return resp, err
}

func (c *ComposeConfigClient) Delete(container *ComposeConfig) error {
	return c.rancherClient.doResourceDelete(COMPOSE_CONFIG_TYPE, &container.Resource)
}
