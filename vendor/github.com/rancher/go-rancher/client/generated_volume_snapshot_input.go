package client

const (
	VOLUME_SNAPSHOT_INPUT_TYPE = "volumeSnapshotInput"
)

type VolumeSnapshotInput struct {
	Resource

	Name string `json:"name,omitempty" yaml:"name,omitempty"`
}

type VolumeSnapshotInputCollection struct {
	Collection
	Data []VolumeSnapshotInput `json:"data,omitempty"`
}

type VolumeSnapshotInputClient struct {
	rancherClient *RancherClient
}

type VolumeSnapshotInputOperations interface {
	List(opts *ListOpts) (*VolumeSnapshotInputCollection, error)
	Create(opts *VolumeSnapshotInput) (*VolumeSnapshotInput, error)
	Update(existing *VolumeSnapshotInput, updates interface{}) (*VolumeSnapshotInput, error)
	ById(id string) (*VolumeSnapshotInput, error)
	Delete(container *VolumeSnapshotInput) error
}

func newVolumeSnapshotInputClient(rancherClient *RancherClient) *VolumeSnapshotInputClient {
	return &VolumeSnapshotInputClient{
		rancherClient: rancherClient,
	}
}

func (c *VolumeSnapshotInputClient) Create(container *VolumeSnapshotInput) (*VolumeSnapshotInput, error) {
	resp := &VolumeSnapshotInput{}
	err := c.rancherClient.doCreate(VOLUME_SNAPSHOT_INPUT_TYPE, container, resp)
	return resp, err
}

func (c *VolumeSnapshotInputClient) Update(existing *VolumeSnapshotInput, updates interface{}) (*VolumeSnapshotInput, error) {
	resp := &VolumeSnapshotInput{}
	err := c.rancherClient.doUpdate(VOLUME_SNAPSHOT_INPUT_TYPE, &existing.Resource, updates, resp)
	return resp, err
}

func (c *VolumeSnapshotInputClient) List(opts *ListOpts) (*VolumeSnapshotInputCollection, error) {
	resp := &VolumeSnapshotInputCollection{}
	err := c.rancherClient.doList(VOLUME_SNAPSHOT_INPUT_TYPE, opts, resp)
	return resp, err
}

func (c *VolumeSnapshotInputClient) ById(id string) (*VolumeSnapshotInput, error) {
	resp := &VolumeSnapshotInput{}
	err := c.rancherClient.doById(VOLUME_SNAPSHOT_INPUT_TYPE, id, resp)
	if apiError, ok := err.(*ApiError); ok {
		if apiError.StatusCode == 404 {
			return nil, nil
		}
	}
	return resp, err
}

func (c *VolumeSnapshotInputClient) Delete(container *VolumeSnapshotInput) error {
	return c.rancherClient.doResourceDelete(VOLUME_SNAPSHOT_INPUT_TYPE, &container.Resource)
}
