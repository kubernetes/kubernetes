package client

const (
	SNAPSHOT_BACKUP_INPUT_TYPE = "snapshotBackupInput"
)

type SnapshotBackupInput struct {
	Resource

	BackupTargetId string `json:"backupTargetId,omitempty" yaml:"backup_target_id,omitempty"`

	Data map[string]interface{} `json:"data,omitempty" yaml:"data,omitempty"`

	Name string `json:"name,omitempty" yaml:"name,omitempty"`

	RemoveTime string `json:"removeTime,omitempty" yaml:"remove_time,omitempty"`
}

type SnapshotBackupInputCollection struct {
	Collection
	Data []SnapshotBackupInput `json:"data,omitempty"`
}

type SnapshotBackupInputClient struct {
	rancherClient *RancherClient
}

type SnapshotBackupInputOperations interface {
	List(opts *ListOpts) (*SnapshotBackupInputCollection, error)
	Create(opts *SnapshotBackupInput) (*SnapshotBackupInput, error)
	Update(existing *SnapshotBackupInput, updates interface{}) (*SnapshotBackupInput, error)
	ById(id string) (*SnapshotBackupInput, error)
	Delete(container *SnapshotBackupInput) error

	ActionCreate(*SnapshotBackupInput) (*Backup, error)

	ActionRemove(*SnapshotBackupInput) (*Backup, error)
}

func newSnapshotBackupInputClient(rancherClient *RancherClient) *SnapshotBackupInputClient {
	return &SnapshotBackupInputClient{
		rancherClient: rancherClient,
	}
}

func (c *SnapshotBackupInputClient) Create(container *SnapshotBackupInput) (*SnapshotBackupInput, error) {
	resp := &SnapshotBackupInput{}
	err := c.rancherClient.doCreate(SNAPSHOT_BACKUP_INPUT_TYPE, container, resp)
	return resp, err
}

func (c *SnapshotBackupInputClient) Update(existing *SnapshotBackupInput, updates interface{}) (*SnapshotBackupInput, error) {
	resp := &SnapshotBackupInput{}
	err := c.rancherClient.doUpdate(SNAPSHOT_BACKUP_INPUT_TYPE, &existing.Resource, updates, resp)
	return resp, err
}

func (c *SnapshotBackupInputClient) List(opts *ListOpts) (*SnapshotBackupInputCollection, error) {
	resp := &SnapshotBackupInputCollection{}
	err := c.rancherClient.doList(SNAPSHOT_BACKUP_INPUT_TYPE, opts, resp)
	return resp, err
}

func (c *SnapshotBackupInputClient) ById(id string) (*SnapshotBackupInput, error) {
	resp := &SnapshotBackupInput{}
	err := c.rancherClient.doById(SNAPSHOT_BACKUP_INPUT_TYPE, id, resp)
	if apiError, ok := err.(*ApiError); ok {
		if apiError.StatusCode == 404 {
			return nil, nil
		}
	}
	return resp, err
}

func (c *SnapshotBackupInputClient) Delete(container *SnapshotBackupInput) error {
	return c.rancherClient.doResourceDelete(SNAPSHOT_BACKUP_INPUT_TYPE, &container.Resource)
}

func (c *SnapshotBackupInputClient) ActionCreate(resource *SnapshotBackupInput) (*Backup, error) {

	resp := &Backup{}

	err := c.rancherClient.doAction(SNAPSHOT_BACKUP_INPUT_TYPE, "create", &resource.Resource, nil, resp)

	return resp, err
}

func (c *SnapshotBackupInputClient) ActionRemove(resource *SnapshotBackupInput) (*Backup, error) {

	resp := &Backup{}

	err := c.rancherClient.doAction(SNAPSHOT_BACKUP_INPUT_TYPE, "remove", &resource.Resource, nil, resp)

	return resp, err
}
