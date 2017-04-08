package client

const (
	RESTORE_FROM_BACKUP_INPUT_TYPE = "restoreFromBackupInput"
)

type RestoreFromBackupInput struct {
	Resource

	BackupId string `json:"backupId,omitempty" yaml:"backup_id,omitempty"`
}

type RestoreFromBackupInputCollection struct {
	Collection
	Data []RestoreFromBackupInput `json:"data,omitempty"`
}

type RestoreFromBackupInputClient struct {
	rancherClient *RancherClient
}

type RestoreFromBackupInputOperations interface {
	List(opts *ListOpts) (*RestoreFromBackupInputCollection, error)
	Create(opts *RestoreFromBackupInput) (*RestoreFromBackupInput, error)
	Update(existing *RestoreFromBackupInput, updates interface{}) (*RestoreFromBackupInput, error)
	ById(id string) (*RestoreFromBackupInput, error)
	Delete(container *RestoreFromBackupInput) error
}

func newRestoreFromBackupInputClient(rancherClient *RancherClient) *RestoreFromBackupInputClient {
	return &RestoreFromBackupInputClient{
		rancherClient: rancherClient,
	}
}

func (c *RestoreFromBackupInputClient) Create(container *RestoreFromBackupInput) (*RestoreFromBackupInput, error) {
	resp := &RestoreFromBackupInput{}
	err := c.rancherClient.doCreate(RESTORE_FROM_BACKUP_INPUT_TYPE, container, resp)
	return resp, err
}

func (c *RestoreFromBackupInputClient) Update(existing *RestoreFromBackupInput, updates interface{}) (*RestoreFromBackupInput, error) {
	resp := &RestoreFromBackupInput{}
	err := c.rancherClient.doUpdate(RESTORE_FROM_BACKUP_INPUT_TYPE, &existing.Resource, updates, resp)
	return resp, err
}

func (c *RestoreFromBackupInputClient) List(opts *ListOpts) (*RestoreFromBackupInputCollection, error) {
	resp := &RestoreFromBackupInputCollection{}
	err := c.rancherClient.doList(RESTORE_FROM_BACKUP_INPUT_TYPE, opts, resp)
	return resp, err
}

func (c *RestoreFromBackupInputClient) ById(id string) (*RestoreFromBackupInput, error) {
	resp := &RestoreFromBackupInput{}
	err := c.rancherClient.doById(RESTORE_FROM_BACKUP_INPUT_TYPE, id, resp)
	if apiError, ok := err.(*ApiError); ok {
		if apiError.StatusCode == 404 {
			return nil, nil
		}
	}
	return resp, err
}

func (c *RestoreFromBackupInputClient) Delete(container *RestoreFromBackupInput) error {
	return c.rancherClient.doResourceDelete(RESTORE_FROM_BACKUP_INPUT_TYPE, &container.Resource)
}
