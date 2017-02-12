package client

const (
	MOUNT_TYPE = "mount"
)

type Mount struct {
	Resource

	AccountId string `json:"accountId,omitempty" yaml:"account_id,omitempty"`

	Created string `json:"created,omitempty" yaml:"created,omitempty"`

	Data map[string]interface{} `json:"data,omitempty" yaml:"data,omitempty"`

	Description string `json:"description,omitempty" yaml:"description,omitempty"`

	InstanceId string `json:"instanceId,omitempty" yaml:"instance_id,omitempty"`

	Kind string `json:"kind,omitempty" yaml:"kind,omitempty"`

	Name string `json:"name,omitempty" yaml:"name,omitempty"`

	Path string `json:"path,omitempty" yaml:"path,omitempty"`

	Permissions string `json:"permissions,omitempty" yaml:"permissions,omitempty"`

	RemoveTime string `json:"removeTime,omitempty" yaml:"remove_time,omitempty"`

	Removed string `json:"removed,omitempty" yaml:"removed,omitempty"`

	State string `json:"state,omitempty" yaml:"state,omitempty"`

	Transitioning string `json:"transitioning,omitempty" yaml:"transitioning,omitempty"`

	TransitioningMessage string `json:"transitioningMessage,omitempty" yaml:"transitioning_message,omitempty"`

	TransitioningProgress int64 `json:"transitioningProgress,omitempty" yaml:"transitioning_progress,omitempty"`

	Uuid string `json:"uuid,omitempty" yaml:"uuid,omitempty"`

	VolumeId string `json:"volumeId,omitempty" yaml:"volume_id,omitempty"`
}

type MountCollection struct {
	Collection
	Data []Mount `json:"data,omitempty"`
}

type MountClient struct {
	rancherClient *RancherClient
}

type MountOperations interface {
	List(opts *ListOpts) (*MountCollection, error)
	Create(opts *Mount) (*Mount, error)
	Update(existing *Mount, updates interface{}) (*Mount, error)
	ById(id string) (*Mount, error)
	Delete(container *Mount) error

	ActionCreate(*Mount) (*Mount, error)

	ActionDeactivate(*Mount) (*Mount, error)

	ActionRemove(*Mount) (*Mount, error)
}

func newMountClient(rancherClient *RancherClient) *MountClient {
	return &MountClient{
		rancherClient: rancherClient,
	}
}

func (c *MountClient) Create(container *Mount) (*Mount, error) {
	resp := &Mount{}
	err := c.rancherClient.doCreate(MOUNT_TYPE, container, resp)
	return resp, err
}

func (c *MountClient) Update(existing *Mount, updates interface{}) (*Mount, error) {
	resp := &Mount{}
	err := c.rancherClient.doUpdate(MOUNT_TYPE, &existing.Resource, updates, resp)
	return resp, err
}

func (c *MountClient) List(opts *ListOpts) (*MountCollection, error) {
	resp := &MountCollection{}
	err := c.rancherClient.doList(MOUNT_TYPE, opts, resp)
	return resp, err
}

func (c *MountClient) ById(id string) (*Mount, error) {
	resp := &Mount{}
	err := c.rancherClient.doById(MOUNT_TYPE, id, resp)
	if apiError, ok := err.(*ApiError); ok {
		if apiError.StatusCode == 404 {
			return nil, nil
		}
	}
	return resp, err
}

func (c *MountClient) Delete(container *Mount) error {
	return c.rancherClient.doResourceDelete(MOUNT_TYPE, &container.Resource)
}

func (c *MountClient) ActionCreate(resource *Mount) (*Mount, error) {

	resp := &Mount{}

	err := c.rancherClient.doAction(MOUNT_TYPE, "create", &resource.Resource, nil, resp)

	return resp, err
}

func (c *MountClient) ActionDeactivate(resource *Mount) (*Mount, error) {

	resp := &Mount{}

	err := c.rancherClient.doAction(MOUNT_TYPE, "deactivate", &resource.Resource, nil, resp)

	return resp, err
}

func (c *MountClient) ActionRemove(resource *Mount) (*Mount, error) {

	resp := &Mount{}

	err := c.rancherClient.doAction(MOUNT_TYPE, "remove", &resource.Resource, nil, resp)

	return resp, err
}
