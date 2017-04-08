package client

const (
	VOLUME_TYPE = "volume"
)

type Volume struct {
	Resource

	AccessMode string `json:"accessMode,omitempty" yaml:"access_mode,omitempty"`

	AccountId string `json:"accountId,omitempty" yaml:"account_id,omitempty"`

	Created string `json:"created,omitempty" yaml:"created,omitempty"`

	Data map[string]interface{} `json:"data,omitempty" yaml:"data,omitempty"`

	Description string `json:"description,omitempty" yaml:"description,omitempty"`

	Driver string `json:"driver,omitempty" yaml:"driver,omitempty"`

	DriverOpts map[string]interface{} `json:"driverOpts,omitempty" yaml:"driver_opts,omitempty"`

	ExternalId string `json:"externalId,omitempty" yaml:"external_id,omitempty"`

	ImageId string `json:"imageId,omitempty" yaml:"image_id,omitempty"`

	InstanceId string `json:"instanceId,omitempty" yaml:"instance_id,omitempty"`

	IsHostPath bool `json:"isHostPath,omitempty" yaml:"is_host_path,omitempty"`

	Kind string `json:"kind,omitempty" yaml:"kind,omitempty"`

	Name string `json:"name,omitempty" yaml:"name,omitempty"`

	RemoveTime string `json:"removeTime,omitempty" yaml:"remove_time,omitempty"`

	Removed string `json:"removed,omitempty" yaml:"removed,omitempty"`

	State string `json:"state,omitempty" yaml:"state,omitempty"`

	Transitioning string `json:"transitioning,omitempty" yaml:"transitioning,omitempty"`

	TransitioningMessage string `json:"transitioningMessage,omitempty" yaml:"transitioning_message,omitempty"`

	TransitioningProgress int64 `json:"transitioningProgress,omitempty" yaml:"transitioning_progress,omitempty"`

	Uri string `json:"uri,omitempty" yaml:"uri,omitempty"`

	Uuid string `json:"uuid,omitempty" yaml:"uuid,omitempty"`
}

type VolumeCollection struct {
	Collection
	Data []Volume `json:"data,omitempty"`
}

type VolumeClient struct {
	rancherClient *RancherClient
}

type VolumeOperations interface {
	List(opts *ListOpts) (*VolumeCollection, error)
	Create(opts *Volume) (*Volume, error)
	Update(existing *Volume, updates interface{}) (*Volume, error)
	ById(id string) (*Volume, error)
	Delete(container *Volume) error

	ActionActivate(*Volume) (*Volume, error)

	ActionAllocate(*Volume) (*Volume, error)

	ActionCreate(*Volume) (*Volume, error)

	ActionDeallocate(*Volume) (*Volume, error)

	ActionPurge(*Volume) (*Volume, error)

	ActionRemove(*Volume) (*Volume, error)

	ActionRestore(*Volume) (*Volume, error)

	ActionRestorefrombackup(*Volume, *RestoreFromBackupInput) (*Volume, error)

	ActionReverttosnapshot(*Volume, *RevertToSnapshotInput) (*Volume, error)

	ActionSnapshot(*Volume, *VolumeSnapshotInput) (*Snapshot, error)

	ActionUpdate(*Volume) (*Volume, error)
}

func newVolumeClient(rancherClient *RancherClient) *VolumeClient {
	return &VolumeClient{
		rancherClient: rancherClient,
	}
}

func (c *VolumeClient) Create(container *Volume) (*Volume, error) {
	resp := &Volume{}
	err := c.rancherClient.doCreate(VOLUME_TYPE, container, resp)
	return resp, err
}

func (c *VolumeClient) Update(existing *Volume, updates interface{}) (*Volume, error) {
	resp := &Volume{}
	err := c.rancherClient.doUpdate(VOLUME_TYPE, &existing.Resource, updates, resp)
	return resp, err
}

func (c *VolumeClient) List(opts *ListOpts) (*VolumeCollection, error) {
	resp := &VolumeCollection{}
	err := c.rancherClient.doList(VOLUME_TYPE, opts, resp)
	return resp, err
}

func (c *VolumeClient) ById(id string) (*Volume, error) {
	resp := &Volume{}
	err := c.rancherClient.doById(VOLUME_TYPE, id, resp)
	if apiError, ok := err.(*ApiError); ok {
		if apiError.StatusCode == 404 {
			return nil, nil
		}
	}
	return resp, err
}

func (c *VolumeClient) Delete(container *Volume) error {
	return c.rancherClient.doResourceDelete(VOLUME_TYPE, &container.Resource)
}

func (c *VolumeClient) ActionActivate(resource *Volume) (*Volume, error) {

	resp := &Volume{}

	err := c.rancherClient.doAction(VOLUME_TYPE, "activate", &resource.Resource, nil, resp)

	return resp, err
}

func (c *VolumeClient) ActionAllocate(resource *Volume) (*Volume, error) {

	resp := &Volume{}

	err := c.rancherClient.doAction(VOLUME_TYPE, "allocate", &resource.Resource, nil, resp)

	return resp, err
}

func (c *VolumeClient) ActionCreate(resource *Volume) (*Volume, error) {

	resp := &Volume{}

	err := c.rancherClient.doAction(VOLUME_TYPE, "create", &resource.Resource, nil, resp)

	return resp, err
}

func (c *VolumeClient) ActionDeallocate(resource *Volume) (*Volume, error) {

	resp := &Volume{}

	err := c.rancherClient.doAction(VOLUME_TYPE, "deallocate", &resource.Resource, nil, resp)

	return resp, err
}

func (c *VolumeClient) ActionPurge(resource *Volume) (*Volume, error) {

	resp := &Volume{}

	err := c.rancherClient.doAction(VOLUME_TYPE, "purge", &resource.Resource, nil, resp)

	return resp, err
}

func (c *VolumeClient) ActionRemove(resource *Volume) (*Volume, error) {

	resp := &Volume{}

	err := c.rancherClient.doAction(VOLUME_TYPE, "remove", &resource.Resource, nil, resp)

	return resp, err
}

func (c *VolumeClient) ActionRestore(resource *Volume) (*Volume, error) {

	resp := &Volume{}

	err := c.rancherClient.doAction(VOLUME_TYPE, "restore", &resource.Resource, nil, resp)

	return resp, err
}

func (c *VolumeClient) ActionRestorefrombackup(resource *Volume, input *RestoreFromBackupInput) (*Volume, error) {

	resp := &Volume{}

	err := c.rancherClient.doAction(VOLUME_TYPE, "restorefrombackup", &resource.Resource, input, resp)

	return resp, err
}

func (c *VolumeClient) ActionReverttosnapshot(resource *Volume, input *RevertToSnapshotInput) (*Volume, error) {

	resp := &Volume{}

	err := c.rancherClient.doAction(VOLUME_TYPE, "reverttosnapshot", &resource.Resource, input, resp)

	return resp, err
}

func (c *VolumeClient) ActionSnapshot(resource *Volume, input *VolumeSnapshotInput) (*Snapshot, error) {

	resp := &Snapshot{}

	err := c.rancherClient.doAction(VOLUME_TYPE, "snapshot", &resource.Resource, input, resp)

	return resp, err
}

func (c *VolumeClient) ActionUpdate(resource *Volume) (*Volume, error) {

	resp := &Volume{}

	err := c.rancherClient.doAction(VOLUME_TYPE, "update", &resource.Resource, nil, resp)

	return resp, err
}
