package client

const (
	KUBERNETES_SERVICE_TYPE = "kubernetesService"
)

type KubernetesService struct {
	Resource

	AccountId string `json:"accountId,omitempty" yaml:"account_id,omitempty"`

	Created string `json:"created,omitempty" yaml:"created,omitempty"`

	Data map[string]interface{} `json:"data,omitempty" yaml:"data,omitempty"`

	Description string `json:"description,omitempty" yaml:"description,omitempty"`

	EnvironmentId string `json:"environmentId,omitempty" yaml:"environment_id,omitempty"`

	ExternalId string `json:"externalId,omitempty" yaml:"external_id,omitempty"`

	HealthState string `json:"healthState,omitempty" yaml:"health_state,omitempty"`

	Kind string `json:"kind,omitempty" yaml:"kind,omitempty"`

	Name string `json:"name,omitempty" yaml:"name,omitempty"`

	RemoveTime string `json:"removeTime,omitempty" yaml:"remove_time,omitempty"`

	Removed string `json:"removed,omitempty" yaml:"removed,omitempty"`

	SelectorContainer string `json:"selectorContainer,omitempty" yaml:"selector_container,omitempty"`

	State string `json:"state,omitempty" yaml:"state,omitempty"`

	Template interface{} `json:"template,omitempty" yaml:"template,omitempty"`

	Transitioning string `json:"transitioning,omitempty" yaml:"transitioning,omitempty"`

	TransitioningMessage string `json:"transitioningMessage,omitempty" yaml:"transitioning_message,omitempty"`

	TransitioningProgress int64 `json:"transitioningProgress,omitempty" yaml:"transitioning_progress,omitempty"`

	Uuid string `json:"uuid,omitempty" yaml:"uuid,omitempty"`

	Vip string `json:"vip,omitempty" yaml:"vip,omitempty"`
}

type KubernetesServiceCollection struct {
	Collection
	Data []KubernetesService `json:"data,omitempty"`
}

type KubernetesServiceClient struct {
	rancherClient *RancherClient
}

type KubernetesServiceOperations interface {
	List(opts *ListOpts) (*KubernetesServiceCollection, error)
	Create(opts *KubernetesService) (*KubernetesService, error)
	Update(existing *KubernetesService, updates interface{}) (*KubernetesService, error)
	ById(id string) (*KubernetesService, error)
	Delete(container *KubernetesService) error

	ActionActivate(*KubernetesService) (*Service, error)

	ActionAddservicelink(*KubernetesService, *AddRemoveServiceLinkInput) (*Service, error)

	ActionCancelrollback(*KubernetesService) (*Service, error)

	ActionCancelupgrade(*KubernetesService) (*Service, error)

	ActionCreate(*KubernetesService) (*Service, error)

	ActionDeactivate(*KubernetesService) (*Service, error)

	ActionFinishupgrade(*KubernetesService) (*Service, error)

	ActionRemove(*KubernetesService) (*Service, error)

	ActionRemoveservicelink(*KubernetesService, *AddRemoveServiceLinkInput) (*Service, error)

	ActionRestart(*KubernetesService, *ServiceRestart) (*Service, error)

	ActionRollback(*KubernetesService) (*Service, error)

	ActionSetservicelinks(*KubernetesService, *SetServiceLinksInput) (*Service, error)

	ActionUpdate(*KubernetesService) (*Service, error)

	ActionUpgrade(*KubernetesService, *ServiceUpgrade) (*Service, error)
}

func newKubernetesServiceClient(rancherClient *RancherClient) *KubernetesServiceClient {
	return &KubernetesServiceClient{
		rancherClient: rancherClient,
	}
}

func (c *KubernetesServiceClient) Create(container *KubernetesService) (*KubernetesService, error) {
	resp := &KubernetesService{}
	err := c.rancherClient.doCreate(KUBERNETES_SERVICE_TYPE, container, resp)
	return resp, err
}

func (c *KubernetesServiceClient) Update(existing *KubernetesService, updates interface{}) (*KubernetesService, error) {
	resp := &KubernetesService{}
	err := c.rancherClient.doUpdate(KUBERNETES_SERVICE_TYPE, &existing.Resource, updates, resp)
	return resp, err
}

func (c *KubernetesServiceClient) List(opts *ListOpts) (*KubernetesServiceCollection, error) {
	resp := &KubernetesServiceCollection{}
	err := c.rancherClient.doList(KUBERNETES_SERVICE_TYPE, opts, resp)
	return resp, err
}

func (c *KubernetesServiceClient) ById(id string) (*KubernetesService, error) {
	resp := &KubernetesService{}
	err := c.rancherClient.doById(KUBERNETES_SERVICE_TYPE, id, resp)
	if apiError, ok := err.(*ApiError); ok {
		if apiError.StatusCode == 404 {
			return nil, nil
		}
	}
	return resp, err
}

func (c *KubernetesServiceClient) Delete(container *KubernetesService) error {
	return c.rancherClient.doResourceDelete(KUBERNETES_SERVICE_TYPE, &container.Resource)
}

func (c *KubernetesServiceClient) ActionActivate(resource *KubernetesService) (*Service, error) {

	resp := &Service{}

	err := c.rancherClient.doAction(KUBERNETES_SERVICE_TYPE, "activate", &resource.Resource, nil, resp)

	return resp, err
}

func (c *KubernetesServiceClient) ActionAddservicelink(resource *KubernetesService, input *AddRemoveServiceLinkInput) (*Service, error) {

	resp := &Service{}

	err := c.rancherClient.doAction(KUBERNETES_SERVICE_TYPE, "addservicelink", &resource.Resource, input, resp)

	return resp, err
}

func (c *KubernetesServiceClient) ActionCancelrollback(resource *KubernetesService) (*Service, error) {

	resp := &Service{}

	err := c.rancherClient.doAction(KUBERNETES_SERVICE_TYPE, "cancelrollback", &resource.Resource, nil, resp)

	return resp, err
}

func (c *KubernetesServiceClient) ActionCancelupgrade(resource *KubernetesService) (*Service, error) {

	resp := &Service{}

	err := c.rancherClient.doAction(KUBERNETES_SERVICE_TYPE, "cancelupgrade", &resource.Resource, nil, resp)

	return resp, err
}

func (c *KubernetesServiceClient) ActionCreate(resource *KubernetesService) (*Service, error) {

	resp := &Service{}

	err := c.rancherClient.doAction(KUBERNETES_SERVICE_TYPE, "create", &resource.Resource, nil, resp)

	return resp, err
}

func (c *KubernetesServiceClient) ActionDeactivate(resource *KubernetesService) (*Service, error) {

	resp := &Service{}

	err := c.rancherClient.doAction(KUBERNETES_SERVICE_TYPE, "deactivate", &resource.Resource, nil, resp)

	return resp, err
}

func (c *KubernetesServiceClient) ActionFinishupgrade(resource *KubernetesService) (*Service, error) {

	resp := &Service{}

	err := c.rancherClient.doAction(KUBERNETES_SERVICE_TYPE, "finishupgrade", &resource.Resource, nil, resp)

	return resp, err
}

func (c *KubernetesServiceClient) ActionRemove(resource *KubernetesService) (*Service, error) {

	resp := &Service{}

	err := c.rancherClient.doAction(KUBERNETES_SERVICE_TYPE, "remove", &resource.Resource, nil, resp)

	return resp, err
}

func (c *KubernetesServiceClient) ActionRemoveservicelink(resource *KubernetesService, input *AddRemoveServiceLinkInput) (*Service, error) {

	resp := &Service{}

	err := c.rancherClient.doAction(KUBERNETES_SERVICE_TYPE, "removeservicelink", &resource.Resource, input, resp)

	return resp, err
}

func (c *KubernetesServiceClient) ActionRestart(resource *KubernetesService, input *ServiceRestart) (*Service, error) {

	resp := &Service{}

	err := c.rancherClient.doAction(KUBERNETES_SERVICE_TYPE, "restart", &resource.Resource, input, resp)

	return resp, err
}

func (c *KubernetesServiceClient) ActionRollback(resource *KubernetesService) (*Service, error) {

	resp := &Service{}

	err := c.rancherClient.doAction(KUBERNETES_SERVICE_TYPE, "rollback", &resource.Resource, nil, resp)

	return resp, err
}

func (c *KubernetesServiceClient) ActionSetservicelinks(resource *KubernetesService, input *SetServiceLinksInput) (*Service, error) {

	resp := &Service{}

	err := c.rancherClient.doAction(KUBERNETES_SERVICE_TYPE, "setservicelinks", &resource.Resource, input, resp)

	return resp, err
}

func (c *KubernetesServiceClient) ActionUpdate(resource *KubernetesService) (*Service, error) {

	resp := &Service{}

	err := c.rancherClient.doAction(KUBERNETES_SERVICE_TYPE, "update", &resource.Resource, nil, resp)

	return resp, err
}

func (c *KubernetesServiceClient) ActionUpgrade(resource *KubernetesService, input *ServiceUpgrade) (*Service, error) {

	resp := &Service{}

	err := c.rancherClient.doAction(KUBERNETES_SERVICE_TYPE, "upgrade", &resource.Resource, input, resp)

	return resp, err
}
