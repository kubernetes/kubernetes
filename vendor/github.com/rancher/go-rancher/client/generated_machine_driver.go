package client

const (
	MACHINE_DRIVER_TYPE = "machineDriver"
)

type MachineDriver struct {
	Resource

	ActivateOnCreate bool `json:"activateOnCreate,omitempty" yaml:"activate_on_create,omitempty"`

	Builtin bool `json:"builtin,omitempty" yaml:"builtin,omitempty"`

	Checksum string `json:"checksum,omitempty" yaml:"checksum,omitempty"`

	Created string `json:"created,omitempty" yaml:"created,omitempty"`

	Data map[string]interface{} `json:"data,omitempty" yaml:"data,omitempty"`

	DefaultActive bool `json:"defaultActive,omitempty" yaml:"default_active,omitempty"`

	Description string `json:"description,omitempty" yaml:"description,omitempty"`

	ExternalId string `json:"externalId,omitempty" yaml:"external_id,omitempty"`

	Kind string `json:"kind,omitempty" yaml:"kind,omitempty"`

	Name string `json:"name,omitempty" yaml:"name,omitempty"`

	RemoveTime string `json:"removeTime,omitempty" yaml:"remove_time,omitempty"`

	Removed string `json:"removed,omitempty" yaml:"removed,omitempty"`

	State string `json:"state,omitempty" yaml:"state,omitempty"`

	Transitioning string `json:"transitioning,omitempty" yaml:"transitioning,omitempty"`

	TransitioningMessage string `json:"transitioningMessage,omitempty" yaml:"transitioning_message,omitempty"`

	TransitioningProgress int64 `json:"transitioningProgress,omitempty" yaml:"transitioning_progress,omitempty"`

	UiUrl string `json:"uiUrl,omitempty" yaml:"ui_url,omitempty"`

	Url string `json:"url,omitempty" yaml:"url,omitempty"`

	Uuid string `json:"uuid,omitempty" yaml:"uuid,omitempty"`
}

type MachineDriverCollection struct {
	Collection
	Data []MachineDriver `json:"data,omitempty"`
}

type MachineDriverClient struct {
	rancherClient *RancherClient
}

type MachineDriverOperations interface {
	List(opts *ListOpts) (*MachineDriverCollection, error)
	Create(opts *MachineDriver) (*MachineDriver, error)
	Update(existing *MachineDriver, updates interface{}) (*MachineDriver, error)
	ById(id string) (*MachineDriver, error)
	Delete(container *MachineDriver) error

	ActionActivate(*MachineDriver) (*MachineDriver, error)

	ActionDeactivate(*MachineDriver) (*MachineDriver, error)

	ActionError(*MachineDriver) (*MachineDriver, error)

	ActionReactivate(*MachineDriver) (*MachineDriver, error)

	ActionRemove(*MachineDriver) (*MachineDriver, error)

	ActionUpdate(*MachineDriver) (*MachineDriver, error)
}

func newMachineDriverClient(rancherClient *RancherClient) *MachineDriverClient {
	return &MachineDriverClient{
		rancherClient: rancherClient,
	}
}

func (c *MachineDriverClient) Create(container *MachineDriver) (*MachineDriver, error) {
	resp := &MachineDriver{}
	err := c.rancherClient.doCreate(MACHINE_DRIVER_TYPE, container, resp)
	return resp, err
}

func (c *MachineDriverClient) Update(existing *MachineDriver, updates interface{}) (*MachineDriver, error) {
	resp := &MachineDriver{}
	err := c.rancherClient.doUpdate(MACHINE_DRIVER_TYPE, &existing.Resource, updates, resp)
	return resp, err
}

func (c *MachineDriverClient) List(opts *ListOpts) (*MachineDriverCollection, error) {
	resp := &MachineDriverCollection{}
	err := c.rancherClient.doList(MACHINE_DRIVER_TYPE, opts, resp)
	return resp, err
}

func (c *MachineDriverClient) ById(id string) (*MachineDriver, error) {
	resp := &MachineDriver{}
	err := c.rancherClient.doById(MACHINE_DRIVER_TYPE, id, resp)
	if apiError, ok := err.(*ApiError); ok {
		if apiError.StatusCode == 404 {
			return nil, nil
		}
	}
	return resp, err
}

func (c *MachineDriverClient) Delete(container *MachineDriver) error {
	return c.rancherClient.doResourceDelete(MACHINE_DRIVER_TYPE, &container.Resource)
}

func (c *MachineDriverClient) ActionActivate(resource *MachineDriver) (*MachineDriver, error) {

	resp := &MachineDriver{}

	err := c.rancherClient.doAction(MACHINE_DRIVER_TYPE, "activate", &resource.Resource, nil, resp)

	return resp, err
}

func (c *MachineDriverClient) ActionDeactivate(resource *MachineDriver) (*MachineDriver, error) {

	resp := &MachineDriver{}

	err := c.rancherClient.doAction(MACHINE_DRIVER_TYPE, "deactivate", &resource.Resource, nil, resp)

	return resp, err
}

func (c *MachineDriverClient) ActionError(resource *MachineDriver) (*MachineDriver, error) {

	resp := &MachineDriver{}

	err := c.rancherClient.doAction(MACHINE_DRIVER_TYPE, "error", &resource.Resource, nil, resp)

	return resp, err
}

func (c *MachineDriverClient) ActionReactivate(resource *MachineDriver) (*MachineDriver, error) {

	resp := &MachineDriver{}

	err := c.rancherClient.doAction(MACHINE_DRIVER_TYPE, "reactivate", &resource.Resource, nil, resp)

	return resp, err
}

func (c *MachineDriverClient) ActionRemove(resource *MachineDriver) (*MachineDriver, error) {

	resp := &MachineDriver{}

	err := c.rancherClient.doAction(MACHINE_DRIVER_TYPE, "remove", &resource.Resource, nil, resp)

	return resp, err
}

func (c *MachineDriverClient) ActionUpdate(resource *MachineDriver) (*MachineDriver, error) {

	resp := &MachineDriver{}

	err := c.rancherClient.doAction(MACHINE_DRIVER_TYPE, "update", &resource.Resource, nil, resp)

	return resp, err
}
