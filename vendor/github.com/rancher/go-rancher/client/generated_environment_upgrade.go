package client

const (
	ENVIRONMENT_UPGRADE_TYPE = "environmentUpgrade"
)

type EnvironmentUpgrade struct {
	Resource

	DockerCompose string `json:"dockerCompose,omitempty" yaml:"docker_compose,omitempty"`

	Environment map[string]interface{} `json:"environment,omitempty" yaml:"environment,omitempty"`

	ExternalId string `json:"externalId,omitempty" yaml:"external_id,omitempty"`

	RancherCompose string `json:"rancherCompose,omitempty" yaml:"rancher_compose,omitempty"`
}

type EnvironmentUpgradeCollection struct {
	Collection
	Data []EnvironmentUpgrade `json:"data,omitempty"`
}

type EnvironmentUpgradeClient struct {
	rancherClient *RancherClient
}

type EnvironmentUpgradeOperations interface {
	List(opts *ListOpts) (*EnvironmentUpgradeCollection, error)
	Create(opts *EnvironmentUpgrade) (*EnvironmentUpgrade, error)
	Update(existing *EnvironmentUpgrade, updates interface{}) (*EnvironmentUpgrade, error)
	ById(id string) (*EnvironmentUpgrade, error)
	Delete(container *EnvironmentUpgrade) error
}

func newEnvironmentUpgradeClient(rancherClient *RancherClient) *EnvironmentUpgradeClient {
	return &EnvironmentUpgradeClient{
		rancherClient: rancherClient,
	}
}

func (c *EnvironmentUpgradeClient) Create(container *EnvironmentUpgrade) (*EnvironmentUpgrade, error) {
	resp := &EnvironmentUpgrade{}
	err := c.rancherClient.doCreate(ENVIRONMENT_UPGRADE_TYPE, container, resp)
	return resp, err
}

func (c *EnvironmentUpgradeClient) Update(existing *EnvironmentUpgrade, updates interface{}) (*EnvironmentUpgrade, error) {
	resp := &EnvironmentUpgrade{}
	err := c.rancherClient.doUpdate(ENVIRONMENT_UPGRADE_TYPE, &existing.Resource, updates, resp)
	return resp, err
}

func (c *EnvironmentUpgradeClient) List(opts *ListOpts) (*EnvironmentUpgradeCollection, error) {
	resp := &EnvironmentUpgradeCollection{}
	err := c.rancherClient.doList(ENVIRONMENT_UPGRADE_TYPE, opts, resp)
	return resp, err
}

func (c *EnvironmentUpgradeClient) ById(id string) (*EnvironmentUpgrade, error) {
	resp := &EnvironmentUpgrade{}
	err := c.rancherClient.doById(ENVIRONMENT_UPGRADE_TYPE, id, resp)
	if apiError, ok := err.(*ApiError); ok {
		if apiError.StatusCode == 404 {
			return nil, nil
		}
	}
	return resp, err
}

func (c *EnvironmentUpgradeClient) Delete(container *EnvironmentUpgrade) error {
	return c.rancherClient.doResourceDelete(ENVIRONMENT_UPGRADE_TYPE, &container.Resource)
}
