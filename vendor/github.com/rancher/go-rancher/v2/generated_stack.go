package client

const (
	STACK_TYPE = "stack"
)

type Stack struct {
	Resource

	AccountId string `json:"accountId,omitempty" yaml:"account_id,omitempty"`

	Binding *Binding `json:"binding,omitempty" yaml:"binding,omitempty"`

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

type StackCollection struct {
	Collection
	Data []Stack `json:"data,omitempty"`
}

type StackClient struct {
	rancherClient *RancherClient
}

type StackOperations interface {
	List(opts *ListOpts) (*StackCollection, error)
	Create(opts *Stack) (*Stack, error)
	Update(existing *Stack, updates interface{}) (*Stack, error)
	ById(id string) (*Stack, error)
	Delete(container *Stack) error

	ActionActivateservices(*Stack) (*Stack, error)

	ActionAddoutputs(*Stack, *AddOutputsInput) (*Stack, error)

	ActionCancelupgrade(*Stack) (*Stack, error)

	ActionCreate(*Stack) (*Stack, error)

	ActionDeactivateservices(*Stack) (*Stack, error)

	ActionError(*Stack) (*Stack, error)

	ActionExportconfig(*Stack, *ComposeConfigInput) (*ComposeConfig, error)

	ActionFinishupgrade(*Stack) (*Stack, error)

	ActionRemove(*Stack) (*Stack, error)

	ActionRollback(*Stack) (*Stack, error)

	ActionUpdate(*Stack) (*Stack, error)

	ActionUpgrade(*Stack, *StackUpgrade) (*Stack, error)
}

func newStackClient(rancherClient *RancherClient) *StackClient {
	return &StackClient{
		rancherClient: rancherClient,
	}
}

func (c *StackClient) Create(container *Stack) (*Stack, error) {
	resp := &Stack{}
	err := c.rancherClient.doCreate(STACK_TYPE, container, resp)
	return resp, err
}

func (c *StackClient) Update(existing *Stack, updates interface{}) (*Stack, error) {
	resp := &Stack{}
	err := c.rancherClient.doUpdate(STACK_TYPE, &existing.Resource, updates, resp)
	return resp, err
}

func (c *StackClient) List(opts *ListOpts) (*StackCollection, error) {
	resp := &StackCollection{}
	err := c.rancherClient.doList(STACK_TYPE, opts, resp)
	return resp, err
}

func (c *StackClient) ById(id string) (*Stack, error) {
	resp := &Stack{}
	err := c.rancherClient.doById(STACK_TYPE, id, resp)
	if apiError, ok := err.(*ApiError); ok {
		if apiError.StatusCode == 404 {
			return nil, nil
		}
	}
	return resp, err
}

func (c *StackClient) Delete(container *Stack) error {
	return c.rancherClient.doResourceDelete(STACK_TYPE, &container.Resource)
}

func (c *StackClient) ActionActivateservices(resource *Stack) (*Stack, error) {

	resp := &Stack{}

	err := c.rancherClient.doAction(STACK_TYPE, "activateservices", &resource.Resource, nil, resp)

	return resp, err
}

func (c *StackClient) ActionAddoutputs(resource *Stack, input *AddOutputsInput) (*Stack, error) {

	resp := &Stack{}

	err := c.rancherClient.doAction(STACK_TYPE, "addoutputs", &resource.Resource, input, resp)

	return resp, err
}

func (c *StackClient) ActionCancelupgrade(resource *Stack) (*Stack, error) {

	resp := &Stack{}

	err := c.rancherClient.doAction(STACK_TYPE, "cancelupgrade", &resource.Resource, nil, resp)

	return resp, err
}

func (c *StackClient) ActionCreate(resource *Stack) (*Stack, error) {

	resp := &Stack{}

	err := c.rancherClient.doAction(STACK_TYPE, "create", &resource.Resource, nil, resp)

	return resp, err
}

func (c *StackClient) ActionDeactivateservices(resource *Stack) (*Stack, error) {

	resp := &Stack{}

	err := c.rancherClient.doAction(STACK_TYPE, "deactivateservices", &resource.Resource, nil, resp)

	return resp, err
}

func (c *StackClient) ActionError(resource *Stack) (*Stack, error) {

	resp := &Stack{}

	err := c.rancherClient.doAction(STACK_TYPE, "error", &resource.Resource, nil, resp)

	return resp, err
}

func (c *StackClient) ActionExportconfig(resource *Stack, input *ComposeConfigInput) (*ComposeConfig, error) {

	resp := &ComposeConfig{}

	err := c.rancherClient.doAction(STACK_TYPE, "exportconfig", &resource.Resource, input, resp)

	return resp, err
}

func (c *StackClient) ActionFinishupgrade(resource *Stack) (*Stack, error) {

	resp := &Stack{}

	err := c.rancherClient.doAction(STACK_TYPE, "finishupgrade", &resource.Resource, nil, resp)

	return resp, err
}

func (c *StackClient) ActionRemove(resource *Stack) (*Stack, error) {

	resp := &Stack{}

	err := c.rancherClient.doAction(STACK_TYPE, "remove", &resource.Resource, nil, resp)

	return resp, err
}

func (c *StackClient) ActionRollback(resource *Stack) (*Stack, error) {

	resp := &Stack{}

	err := c.rancherClient.doAction(STACK_TYPE, "rollback", &resource.Resource, nil, resp)

	return resp, err
}

func (c *StackClient) ActionUpdate(resource *Stack) (*Stack, error) {

	resp := &Stack{}

	err := c.rancherClient.doAction(STACK_TYPE, "update", &resource.Resource, nil, resp)

	return resp, err
}

func (c *StackClient) ActionUpgrade(resource *Stack, input *StackUpgrade) (*Stack, error) {

	resp := &Stack{}

	err := c.rancherClient.doAction(STACK_TYPE, "upgrade", &resource.Resource, input, resp)

	return resp, err
}
