package client

const (
	KUBERNETES_STACK_TYPE = "kubernetesStack"
)

type KubernetesStack struct {
	Resource

	AccountId string `json:"accountId,omitempty" yaml:"account_id,omitempty"`

	Created string `json:"created,omitempty" yaml:"created,omitempty"`

	Data map[string]interface{} `json:"data,omitempty" yaml:"data,omitempty"`

	Description string `json:"description,omitempty" yaml:"description,omitempty"`

	Environment map[string]interface{} `json:"environment,omitempty" yaml:"environment,omitempty"`

	ExternalId string `json:"externalId,omitempty" yaml:"external_id,omitempty"`

	HealthState string `json:"healthState,omitempty" yaml:"health_state,omitempty"`

	Kind string `json:"kind,omitempty" yaml:"kind,omitempty"`

	Name string `json:"name,omitempty" yaml:"name,omitempty"`

	Namespace string `json:"namespace,omitempty" yaml:"namespace,omitempty"`

	PreviousEnvironment map[string]interface{} `json:"previousEnvironment,omitempty" yaml:"previous_environment,omitempty"`

	PreviousExternalId string `json:"previousExternalId,omitempty" yaml:"previous_external_id,omitempty"`

	RemoveTime string `json:"removeTime,omitempty" yaml:"remove_time,omitempty"`

	Removed string `json:"removed,omitempty" yaml:"removed,omitempty"`

	State string `json:"state,omitempty" yaml:"state,omitempty"`

	Templates map[string]interface{} `json:"templates,omitempty" yaml:"templates,omitempty"`

	Transitioning string `json:"transitioning,omitempty" yaml:"transitioning,omitempty"`

	TransitioningMessage string `json:"transitioningMessage,omitempty" yaml:"transitioning_message,omitempty"`

	TransitioningProgress int64 `json:"transitioningProgress,omitempty" yaml:"transitioning_progress,omitempty"`

	Uuid string `json:"uuid,omitempty" yaml:"uuid,omitempty"`
}

type KubernetesStackCollection struct {
	Collection
	Data []KubernetesStack `json:"data,omitempty"`
}

type KubernetesStackClient struct {
	rancherClient *RancherClient
}

type KubernetesStackOperations interface {
	List(opts *ListOpts) (*KubernetesStackCollection, error)
	Create(opts *KubernetesStack) (*KubernetesStack, error)
	Update(existing *KubernetesStack, updates interface{}) (*KubernetesStack, error)
	ById(id string) (*KubernetesStack, error)
	Delete(container *KubernetesStack) error

	ActionCancelrollback(*KubernetesStack) (*Environment, error)

	ActionCancelupgrade(*KubernetesStack) (*Environment, error)

	ActionCreate(*KubernetesStack) (*Environment, error)

	ActionError(*KubernetesStack) (*Environment, error)

	ActionFinishupgrade(*KubernetesStack) (*Environment, error)

	ActionRemove(*KubernetesStack) (*Environment, error)

	ActionRollback(*KubernetesStack) (*Environment, error)

	ActionUpgrade(*KubernetesStack, *KubernetesStackUpgrade) (*KubernetesStack, error)
}

func newKubernetesStackClient(rancherClient *RancherClient) *KubernetesStackClient {
	return &KubernetesStackClient{
		rancherClient: rancherClient,
	}
}

func (c *KubernetesStackClient) Create(container *KubernetesStack) (*KubernetesStack, error) {
	resp := &KubernetesStack{}
	err := c.rancherClient.doCreate(KUBERNETES_STACK_TYPE, container, resp)
	return resp, err
}

func (c *KubernetesStackClient) Update(existing *KubernetesStack, updates interface{}) (*KubernetesStack, error) {
	resp := &KubernetesStack{}
	err := c.rancherClient.doUpdate(KUBERNETES_STACK_TYPE, &existing.Resource, updates, resp)
	return resp, err
}

func (c *KubernetesStackClient) List(opts *ListOpts) (*KubernetesStackCollection, error) {
	resp := &KubernetesStackCollection{}
	err := c.rancherClient.doList(KUBERNETES_STACK_TYPE, opts, resp)
	return resp, err
}

func (c *KubernetesStackClient) ById(id string) (*KubernetesStack, error) {
	resp := &KubernetesStack{}
	err := c.rancherClient.doById(KUBERNETES_STACK_TYPE, id, resp)
	if apiError, ok := err.(*ApiError); ok {
		if apiError.StatusCode == 404 {
			return nil, nil
		}
	}
	return resp, err
}

func (c *KubernetesStackClient) Delete(container *KubernetesStack) error {
	return c.rancherClient.doResourceDelete(KUBERNETES_STACK_TYPE, &container.Resource)
}

func (c *KubernetesStackClient) ActionCancelrollback(resource *KubernetesStack) (*Environment, error) {

	resp := &Environment{}

	err := c.rancherClient.doAction(KUBERNETES_STACK_TYPE, "cancelrollback", &resource.Resource, nil, resp)

	return resp, err
}

func (c *KubernetesStackClient) ActionCancelupgrade(resource *KubernetesStack) (*Environment, error) {

	resp := &Environment{}

	err := c.rancherClient.doAction(KUBERNETES_STACK_TYPE, "cancelupgrade", &resource.Resource, nil, resp)

	return resp, err
}

func (c *KubernetesStackClient) ActionCreate(resource *KubernetesStack) (*Environment, error) {

	resp := &Environment{}

	err := c.rancherClient.doAction(KUBERNETES_STACK_TYPE, "create", &resource.Resource, nil, resp)

	return resp, err
}

func (c *KubernetesStackClient) ActionError(resource *KubernetesStack) (*Environment, error) {

	resp := &Environment{}

	err := c.rancherClient.doAction(KUBERNETES_STACK_TYPE, "error", &resource.Resource, nil, resp)

	return resp, err
}

func (c *KubernetesStackClient) ActionFinishupgrade(resource *KubernetesStack) (*Environment, error) {

	resp := &Environment{}

	err := c.rancherClient.doAction(KUBERNETES_STACK_TYPE, "finishupgrade", &resource.Resource, nil, resp)

	return resp, err
}

func (c *KubernetesStackClient) ActionRemove(resource *KubernetesStack) (*Environment, error) {

	resp := &Environment{}

	err := c.rancherClient.doAction(KUBERNETES_STACK_TYPE, "remove", &resource.Resource, nil, resp)

	return resp, err
}

func (c *KubernetesStackClient) ActionRollback(resource *KubernetesStack) (*Environment, error) {

	resp := &Environment{}

	err := c.rancherClient.doAction(KUBERNETES_STACK_TYPE, "rollback", &resource.Resource, nil, resp)

	return resp, err
}

func (c *KubernetesStackClient) ActionUpgrade(resource *KubernetesStack, input *KubernetesStackUpgrade) (*KubernetesStack, error) {

	resp := &KubernetesStack{}

	err := c.rancherClient.doAction(KUBERNETES_STACK_TYPE, "upgrade", &resource.Resource, input, resp)

	return resp, err
}
