package client

const (
	ENVIRONMENT_TYPE = "environment"
)

type Environment struct {
	Resource

	AccountId string `json:"accountId,omitempty" yaml:"account_id,omitempty"`

	Created string `json:"created,omitempty" yaml:"created,omitempty"`

	Data map[string]interface{} `json:"data,omitempty" yaml:"data,omitempty"`

	Description string `json:"description,omitempty" yaml:"description,omitempty"`

	DockerCompose string `json:"dockerCompose,omitempty" yaml:"docker_compose,omitempty"`

	Environment map[string]interface{} `json:"environment,omitempty" yaml:"environment,omitempty"`

	ExternalId string `json:"externalId,omitempty" yaml:"external_id,omitempty"`

	HealthState string `json:"healthState,omitempty" yaml:"health_state,omitempty"`

	Kind string `json:"kind,omitempty" yaml:"kind,omitempty"`

	Name string `json:"name,omitempty" yaml:"name,omitempty"`

	Outputs map[string]interface{} `json:"outputs,omitempty" yaml:"outputs,omitempty"`

	PreviousEnvironment map[string]interface{} `json:"previousEnvironment,omitempty" yaml:"previous_environment,omitempty"`

	PreviousExternalId string `json:"previousExternalId,omitempty" yaml:"previous_external_id,omitempty"`

	RancherCompose string `json:"rancherCompose,omitempty" yaml:"rancher_compose,omitempty"`

	RemoveTime string `json:"removeTime,omitempty" yaml:"remove_time,omitempty"`

	Removed string `json:"removed,omitempty" yaml:"removed,omitempty"`

	StartOnCreate bool `json:"startOnCreate,omitempty" yaml:"start_on_create,omitempty"`

	State string `json:"state,omitempty" yaml:"state,omitempty"`

	Transitioning string `json:"transitioning,omitempty" yaml:"transitioning,omitempty"`

	TransitioningMessage string `json:"transitioningMessage,omitempty" yaml:"transitioning_message,omitempty"`

	TransitioningProgress int64 `json:"transitioningProgress,omitempty" yaml:"transitioning_progress,omitempty"`

	Uuid string `json:"uuid,omitempty" yaml:"uuid,omitempty"`
}

type EnvironmentCollection struct {
	Collection
	Data []Environment `json:"data,omitempty"`
}

type EnvironmentClient struct {
	rancherClient *RancherClient
}

type EnvironmentOperations interface {
	List(opts *ListOpts) (*EnvironmentCollection, error)
	Create(opts *Environment) (*Environment, error)
	Update(existing *Environment, updates interface{}) (*Environment, error)
	ById(id string) (*Environment, error)
	Delete(container *Environment) error

	ActionActivateservices(*Environment) (*Environment, error)

	ActionAddoutputs(*Environment, *AddOutputsInput) (*Environment, error)

	ActionCancelrollback(*Environment) (*Environment, error)

	ActionCancelupgrade(*Environment) (*Environment, error)

	ActionCreate(*Environment) (*Environment, error)

	ActionDeactivateservices(*Environment) (*Environment, error)

	ActionError(*Environment) (*Environment, error)

	ActionExportconfig(*Environment, *ComposeConfigInput) (*ComposeConfig, error)

	ActionFinishupgrade(*Environment) (*Environment, error)

	ActionRemove(*Environment) (*Environment, error)

	ActionRollback(*Environment) (*Environment, error)

	ActionUpdate(*Environment) (*Environment, error)

	ActionUpgrade(*Environment, *EnvironmentUpgrade) (*Environment, error)
}

func newEnvironmentClient(rancherClient *RancherClient) *EnvironmentClient {
	return &EnvironmentClient{
		rancherClient: rancherClient,
	}
}

func (c *EnvironmentClient) Create(container *Environment) (*Environment, error) {
	resp := &Environment{}
	err := c.rancherClient.doCreate(ENVIRONMENT_TYPE, container, resp)
	return resp, err
}

func (c *EnvironmentClient) Update(existing *Environment, updates interface{}) (*Environment, error) {
	resp := &Environment{}
	err := c.rancherClient.doUpdate(ENVIRONMENT_TYPE, &existing.Resource, updates, resp)
	return resp, err
}

func (c *EnvironmentClient) List(opts *ListOpts) (*EnvironmentCollection, error) {
	resp := &EnvironmentCollection{}
	err := c.rancherClient.doList(ENVIRONMENT_TYPE, opts, resp)
	return resp, err
}

func (c *EnvironmentClient) ById(id string) (*Environment, error) {
	resp := &Environment{}
	err := c.rancherClient.doById(ENVIRONMENT_TYPE, id, resp)
	if apiError, ok := err.(*ApiError); ok {
		if apiError.StatusCode == 404 {
			return nil, nil
		}
	}
	return resp, err
}

func (c *EnvironmentClient) Delete(container *Environment) error {
	return c.rancherClient.doResourceDelete(ENVIRONMENT_TYPE, &container.Resource)
}

func (c *EnvironmentClient) ActionActivateservices(resource *Environment) (*Environment, error) {

	resp := &Environment{}

	err := c.rancherClient.doAction(ENVIRONMENT_TYPE, "activateservices", &resource.Resource, nil, resp)

	return resp, err
}

func (c *EnvironmentClient) ActionAddoutputs(resource *Environment, input *AddOutputsInput) (*Environment, error) {

	resp := &Environment{}

	err := c.rancherClient.doAction(ENVIRONMENT_TYPE, "addoutputs", &resource.Resource, input, resp)

	return resp, err
}

func (c *EnvironmentClient) ActionCancelrollback(resource *Environment) (*Environment, error) {

	resp := &Environment{}

	err := c.rancherClient.doAction(ENVIRONMENT_TYPE, "cancelrollback", &resource.Resource, nil, resp)

	return resp, err
}

func (c *EnvironmentClient) ActionCancelupgrade(resource *Environment) (*Environment, error) {

	resp := &Environment{}

	err := c.rancherClient.doAction(ENVIRONMENT_TYPE, "cancelupgrade", &resource.Resource, nil, resp)

	return resp, err
}

func (c *EnvironmentClient) ActionCreate(resource *Environment) (*Environment, error) {

	resp := &Environment{}

	err := c.rancherClient.doAction(ENVIRONMENT_TYPE, "create", &resource.Resource, nil, resp)

	return resp, err
}

func (c *EnvironmentClient) ActionDeactivateservices(resource *Environment) (*Environment, error) {

	resp := &Environment{}

	err := c.rancherClient.doAction(ENVIRONMENT_TYPE, "deactivateservices", &resource.Resource, nil, resp)

	return resp, err
}

func (c *EnvironmentClient) ActionError(resource *Environment) (*Environment, error) {

	resp := &Environment{}

	err := c.rancherClient.doAction(ENVIRONMENT_TYPE, "error", &resource.Resource, nil, resp)

	return resp, err
}

func (c *EnvironmentClient) ActionExportconfig(resource *Environment, input *ComposeConfigInput) (*ComposeConfig, error) {

	resp := &ComposeConfig{}

	err := c.rancherClient.doAction(ENVIRONMENT_TYPE, "exportconfig", &resource.Resource, input, resp)

	return resp, err
}

func (c *EnvironmentClient) ActionFinishupgrade(resource *Environment) (*Environment, error) {

	resp := &Environment{}

	err := c.rancherClient.doAction(ENVIRONMENT_TYPE, "finishupgrade", &resource.Resource, nil, resp)

	return resp, err
}

func (c *EnvironmentClient) ActionRemove(resource *Environment) (*Environment, error) {

	resp := &Environment{}

	err := c.rancherClient.doAction(ENVIRONMENT_TYPE, "remove", &resource.Resource, nil, resp)

	return resp, err
}

func (c *EnvironmentClient) ActionRollback(resource *Environment) (*Environment, error) {

	resp := &Environment{}

	err := c.rancherClient.doAction(ENVIRONMENT_TYPE, "rollback", &resource.Resource, nil, resp)

	return resp, err
}

func (c *EnvironmentClient) ActionUpdate(resource *Environment) (*Environment, error) {

	resp := &Environment{}

	err := c.rancherClient.doAction(ENVIRONMENT_TYPE, "update", &resource.Resource, nil, resp)

	return resp, err
}

func (c *EnvironmentClient) ActionUpgrade(resource *Environment, input *EnvironmentUpgrade) (*Environment, error) {

	resp := &Environment{}

	err := c.rancherClient.doAction(ENVIRONMENT_TYPE, "upgrade", &resource.Resource, input, resp)

	return resp, err
}
