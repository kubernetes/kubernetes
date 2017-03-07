package client

const (
	PROJECT_TYPE = "project"
)

type Project struct {
	Resource

	AllowSystemRole bool `json:"allowSystemRole,omitempty" yaml:"allow_system_role,omitempty"`

	Created string `json:"created,omitempty" yaml:"created,omitempty"`

	Data map[string]interface{} `json:"data,omitempty" yaml:"data,omitempty"`

	Description string `json:"description,omitempty" yaml:"description,omitempty"`

	Kind string `json:"kind,omitempty" yaml:"kind,omitempty"`

	Kubernetes bool `json:"kubernetes,omitempty" yaml:"kubernetes,omitempty"`

	Members []interface{} `json:"members,omitempty" yaml:"members,omitempty"`

	Mesos bool `json:"mesos,omitempty" yaml:"mesos,omitempty"`

	Name string `json:"name,omitempty" yaml:"name,omitempty"`

	PublicDns bool `json:"publicDns,omitempty" yaml:"public_dns,omitempty"`

	RemoveTime string `json:"removeTime,omitempty" yaml:"remove_time,omitempty"`

	Removed string `json:"removed,omitempty" yaml:"removed,omitempty"`

	ServicesPortRange *ServicesPortRange `json:"servicesPortRange,omitempty" yaml:"services_port_range,omitempty"`

	State string `json:"state,omitempty" yaml:"state,omitempty"`

	Swarm bool `json:"swarm,omitempty" yaml:"swarm,omitempty"`

	Transitioning string `json:"transitioning,omitempty" yaml:"transitioning,omitempty"`

	TransitioningMessage string `json:"transitioningMessage,omitempty" yaml:"transitioning_message,omitempty"`

	TransitioningProgress int64 `json:"transitioningProgress,omitempty" yaml:"transitioning_progress,omitempty"`

	Uuid string `json:"uuid,omitempty" yaml:"uuid,omitempty"`

	VirtualMachine bool `json:"virtualMachine,omitempty" yaml:"virtual_machine,omitempty"`
}

type ProjectCollection struct {
	Collection
	Data []Project `json:"data,omitempty"`
}

type ProjectClient struct {
	rancherClient *RancherClient
}

type ProjectOperations interface {
	List(opts *ListOpts) (*ProjectCollection, error)
	Create(opts *Project) (*Project, error)
	Update(existing *Project, updates interface{}) (*Project, error)
	ById(id string) (*Project, error)
	Delete(container *Project) error

	ActionActivate(*Project) (*Account, error)

	ActionCreate(*Project) (*Account, error)

	ActionDeactivate(*Project) (*Account, error)

	ActionPurge(*Project) (*Account, error)

	ActionRemove(*Project) (*Account, error)

	ActionRestore(*Project) (*Account, error)

	ActionSetmembers(*Project, *SetProjectMembersInput) (*SetProjectMembersInput, error)

	ActionUpdate(*Project) (*Account, error)
}

func newProjectClient(rancherClient *RancherClient) *ProjectClient {
	return &ProjectClient{
		rancherClient: rancherClient,
	}
}

func (c *ProjectClient) Create(container *Project) (*Project, error) {
	resp := &Project{}
	err := c.rancherClient.doCreate(PROJECT_TYPE, container, resp)
	return resp, err
}

func (c *ProjectClient) Update(existing *Project, updates interface{}) (*Project, error) {
	resp := &Project{}
	err := c.rancherClient.doUpdate(PROJECT_TYPE, &existing.Resource, updates, resp)
	return resp, err
}

func (c *ProjectClient) List(opts *ListOpts) (*ProjectCollection, error) {
	resp := &ProjectCollection{}
	err := c.rancherClient.doList(PROJECT_TYPE, opts, resp)
	return resp, err
}

func (c *ProjectClient) ById(id string) (*Project, error) {
	resp := &Project{}
	err := c.rancherClient.doById(PROJECT_TYPE, id, resp)
	if apiError, ok := err.(*ApiError); ok {
		if apiError.StatusCode == 404 {
			return nil, nil
		}
	}
	return resp, err
}

func (c *ProjectClient) Delete(container *Project) error {
	return c.rancherClient.doResourceDelete(PROJECT_TYPE, &container.Resource)
}

func (c *ProjectClient) ActionActivate(resource *Project) (*Account, error) {

	resp := &Account{}

	err := c.rancherClient.doAction(PROJECT_TYPE, "activate", &resource.Resource, nil, resp)

	return resp, err
}

func (c *ProjectClient) ActionCreate(resource *Project) (*Account, error) {

	resp := &Account{}

	err := c.rancherClient.doAction(PROJECT_TYPE, "create", &resource.Resource, nil, resp)

	return resp, err
}

func (c *ProjectClient) ActionDeactivate(resource *Project) (*Account, error) {

	resp := &Account{}

	err := c.rancherClient.doAction(PROJECT_TYPE, "deactivate", &resource.Resource, nil, resp)

	return resp, err
}

func (c *ProjectClient) ActionPurge(resource *Project) (*Account, error) {

	resp := &Account{}

	err := c.rancherClient.doAction(PROJECT_TYPE, "purge", &resource.Resource, nil, resp)

	return resp, err
}

func (c *ProjectClient) ActionRemove(resource *Project) (*Account, error) {

	resp := &Account{}

	err := c.rancherClient.doAction(PROJECT_TYPE, "remove", &resource.Resource, nil, resp)

	return resp, err
}

func (c *ProjectClient) ActionRestore(resource *Project) (*Account, error) {

	resp := &Account{}

	err := c.rancherClient.doAction(PROJECT_TYPE, "restore", &resource.Resource, nil, resp)

	return resp, err
}

func (c *ProjectClient) ActionSetmembers(resource *Project, input *SetProjectMembersInput) (*SetProjectMembersInput, error) {

	resp := &SetProjectMembersInput{}

	err := c.rancherClient.doAction(PROJECT_TYPE, "setmembers", &resource.Resource, input, resp)

	return resp, err
}

func (c *ProjectClient) ActionUpdate(resource *Project) (*Account, error) {

	resp := &Account{}

	err := c.rancherClient.doAction(PROJECT_TYPE, "update", &resource.Resource, nil, resp)

	return resp, err
}
