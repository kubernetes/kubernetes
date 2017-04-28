package client

const (
	EXTERNAL_SERVICE_TYPE = "externalService"
)

type ExternalService struct {
	Resource

	AccountId string `json:"accountId,omitempty" yaml:"account_id,omitempty"`

	Created string `json:"created,omitempty" yaml:"created,omitempty"`

	Data map[string]interface{} `json:"data,omitempty" yaml:"data,omitempty"`

	Description string `json:"description,omitempty" yaml:"description,omitempty"`

	EnvironmentId string `json:"environmentId,omitempty" yaml:"environment_id,omitempty"`

	ExternalId string `json:"externalId,omitempty" yaml:"external_id,omitempty"`

	ExternalIpAddresses []string `json:"externalIpAddresses,omitempty" yaml:"external_ip_addresses,omitempty"`

	Fqdn string `json:"fqdn,omitempty" yaml:"fqdn,omitempty"`

	HealthCheck *InstanceHealthCheck `json:"healthCheck,omitempty" yaml:"health_check,omitempty"`

	HealthState string `json:"healthState,omitempty" yaml:"health_state,omitempty"`

	Hostname string `json:"hostname,omitempty" yaml:"hostname,omitempty"`

	Kind string `json:"kind,omitempty" yaml:"kind,omitempty"`

	LaunchConfig *LaunchConfig `json:"launchConfig,omitempty" yaml:"launch_config,omitempty"`

	Metadata map[string]interface{} `json:"metadata,omitempty" yaml:"metadata,omitempty"`

	Name string `json:"name,omitempty" yaml:"name,omitempty"`

	RemoveTime string `json:"removeTime,omitempty" yaml:"remove_time,omitempty"`

	Removed string `json:"removed,omitempty" yaml:"removed,omitempty"`

	StartOnCreate bool `json:"startOnCreate,omitempty" yaml:"start_on_create,omitempty"`

	State string `json:"state,omitempty" yaml:"state,omitempty"`

	Transitioning string `json:"transitioning,omitempty" yaml:"transitioning,omitempty"`

	TransitioningMessage string `json:"transitioningMessage,omitempty" yaml:"transitioning_message,omitempty"`

	TransitioningProgress int64 `json:"transitioningProgress,omitempty" yaml:"transitioning_progress,omitempty"`

	Upgrade *ServiceUpgrade `json:"upgrade,omitempty" yaml:"upgrade,omitempty"`

	Uuid string `json:"uuid,omitempty" yaml:"uuid,omitempty"`
}

type ExternalServiceCollection struct {
	Collection
	Data []ExternalService `json:"data,omitempty"`
}

type ExternalServiceClient struct {
	rancherClient *RancherClient
}

type ExternalServiceOperations interface {
	List(opts *ListOpts) (*ExternalServiceCollection, error)
	Create(opts *ExternalService) (*ExternalService, error)
	Update(existing *ExternalService, updates interface{}) (*ExternalService, error)
	ById(id string) (*ExternalService, error)
	Delete(container *ExternalService) error

	ActionActivate(*ExternalService) (*Service, error)

	ActionCancelrollback(*ExternalService) (*Service, error)

	ActionCancelupgrade(*ExternalService) (*Service, error)

	ActionCreate(*ExternalService) (*Service, error)

	ActionDeactivate(*ExternalService) (*Service, error)

	ActionFinishupgrade(*ExternalService) (*Service, error)

	ActionRemove(*ExternalService) (*Service, error)

	ActionRestart(*ExternalService, *ServiceRestart) (*Service, error)

	ActionRollback(*ExternalService) (*Service, error)

	ActionUpdate(*ExternalService) (*Service, error)

	ActionUpgrade(*ExternalService, *ServiceUpgrade) (*Service, error)
}

func newExternalServiceClient(rancherClient *RancherClient) *ExternalServiceClient {
	return &ExternalServiceClient{
		rancherClient: rancherClient,
	}
}

func (c *ExternalServiceClient) Create(container *ExternalService) (*ExternalService, error) {
	resp := &ExternalService{}
	err := c.rancherClient.doCreate(EXTERNAL_SERVICE_TYPE, container, resp)
	return resp, err
}

func (c *ExternalServiceClient) Update(existing *ExternalService, updates interface{}) (*ExternalService, error) {
	resp := &ExternalService{}
	err := c.rancherClient.doUpdate(EXTERNAL_SERVICE_TYPE, &existing.Resource, updates, resp)
	return resp, err
}

func (c *ExternalServiceClient) List(opts *ListOpts) (*ExternalServiceCollection, error) {
	resp := &ExternalServiceCollection{}
	err := c.rancherClient.doList(EXTERNAL_SERVICE_TYPE, opts, resp)
	return resp, err
}

func (c *ExternalServiceClient) ById(id string) (*ExternalService, error) {
	resp := &ExternalService{}
	err := c.rancherClient.doById(EXTERNAL_SERVICE_TYPE, id, resp)
	if apiError, ok := err.(*ApiError); ok {
		if apiError.StatusCode == 404 {
			return nil, nil
		}
	}
	return resp, err
}

func (c *ExternalServiceClient) Delete(container *ExternalService) error {
	return c.rancherClient.doResourceDelete(EXTERNAL_SERVICE_TYPE, &container.Resource)
}

func (c *ExternalServiceClient) ActionActivate(resource *ExternalService) (*Service, error) {

	resp := &Service{}

	err := c.rancherClient.doAction(EXTERNAL_SERVICE_TYPE, "activate", &resource.Resource, nil, resp)

	return resp, err
}

func (c *ExternalServiceClient) ActionCancelrollback(resource *ExternalService) (*Service, error) {

	resp := &Service{}

	err := c.rancherClient.doAction(EXTERNAL_SERVICE_TYPE, "cancelrollback", &resource.Resource, nil, resp)

	return resp, err
}

func (c *ExternalServiceClient) ActionCancelupgrade(resource *ExternalService) (*Service, error) {

	resp := &Service{}

	err := c.rancherClient.doAction(EXTERNAL_SERVICE_TYPE, "cancelupgrade", &resource.Resource, nil, resp)

	return resp, err
}

func (c *ExternalServiceClient) ActionCreate(resource *ExternalService) (*Service, error) {

	resp := &Service{}

	err := c.rancherClient.doAction(EXTERNAL_SERVICE_TYPE, "create", &resource.Resource, nil, resp)

	return resp, err
}

func (c *ExternalServiceClient) ActionDeactivate(resource *ExternalService) (*Service, error) {

	resp := &Service{}

	err := c.rancherClient.doAction(EXTERNAL_SERVICE_TYPE, "deactivate", &resource.Resource, nil, resp)

	return resp, err
}

func (c *ExternalServiceClient) ActionFinishupgrade(resource *ExternalService) (*Service, error) {

	resp := &Service{}

	err := c.rancherClient.doAction(EXTERNAL_SERVICE_TYPE, "finishupgrade", &resource.Resource, nil, resp)

	return resp, err
}

func (c *ExternalServiceClient) ActionRemove(resource *ExternalService) (*Service, error) {

	resp := &Service{}

	err := c.rancherClient.doAction(EXTERNAL_SERVICE_TYPE, "remove", &resource.Resource, nil, resp)

	return resp, err
}

func (c *ExternalServiceClient) ActionRestart(resource *ExternalService, input *ServiceRestart) (*Service, error) {

	resp := &Service{}

	err := c.rancherClient.doAction(EXTERNAL_SERVICE_TYPE, "restart", &resource.Resource, input, resp)

	return resp, err
}

func (c *ExternalServiceClient) ActionRollback(resource *ExternalService) (*Service, error) {

	resp := &Service{}

	err := c.rancherClient.doAction(EXTERNAL_SERVICE_TYPE, "rollback", &resource.Resource, nil, resp)

	return resp, err
}

func (c *ExternalServiceClient) ActionUpdate(resource *ExternalService) (*Service, error) {

	resp := &Service{}

	err := c.rancherClient.doAction(EXTERNAL_SERVICE_TYPE, "update", &resource.Resource, nil, resp)

	return resp, err
}

func (c *ExternalServiceClient) ActionUpgrade(resource *ExternalService, input *ServiceUpgrade) (*Service, error) {

	resp := &Service{}

	err := c.rancherClient.doAction(EXTERNAL_SERVICE_TYPE, "upgrade", &resource.Resource, input, resp)

	return resp, err
}
