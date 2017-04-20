package client

const (
	CONFIG_ITEM_STATUS_TYPE = "configItemStatus"
)

type ConfigItemStatus struct {
	Resource

	AccountId string `json:"accountId,omitempty" yaml:"account_id,omitempty"`

	AgentId string `json:"agentId,omitempty" yaml:"agent_id,omitempty"`

	AppliedUpdated string `json:"appliedUpdated,omitempty" yaml:"applied_updated,omitempty"`

	AppliedVersion int64 `json:"appliedVersion,omitempty" yaml:"applied_version,omitempty"`

	Name string `json:"name,omitempty" yaml:"name,omitempty"`

	RequestedUpdated string `json:"requestedUpdated,omitempty" yaml:"requested_updated,omitempty"`

	RequestedVersion int64 `json:"requestedVersion,omitempty" yaml:"requested_version,omitempty"`

	SourceVersion string `json:"sourceVersion,omitempty" yaml:"source_version,omitempty"`
}

type ConfigItemStatusCollection struct {
	Collection
	Data []ConfigItemStatus `json:"data,omitempty"`
}

type ConfigItemStatusClient struct {
	rancherClient *RancherClient
}

type ConfigItemStatusOperations interface {
	List(opts *ListOpts) (*ConfigItemStatusCollection, error)
	Create(opts *ConfigItemStatus) (*ConfigItemStatus, error)
	Update(existing *ConfigItemStatus, updates interface{}) (*ConfigItemStatus, error)
	ById(id string) (*ConfigItemStatus, error)
	Delete(container *ConfigItemStatus) error
}

func newConfigItemStatusClient(rancherClient *RancherClient) *ConfigItemStatusClient {
	return &ConfigItemStatusClient{
		rancherClient: rancherClient,
	}
}

func (c *ConfigItemStatusClient) Create(container *ConfigItemStatus) (*ConfigItemStatus, error) {
	resp := &ConfigItemStatus{}
	err := c.rancherClient.doCreate(CONFIG_ITEM_STATUS_TYPE, container, resp)
	return resp, err
}

func (c *ConfigItemStatusClient) Update(existing *ConfigItemStatus, updates interface{}) (*ConfigItemStatus, error) {
	resp := &ConfigItemStatus{}
	err := c.rancherClient.doUpdate(CONFIG_ITEM_STATUS_TYPE, &existing.Resource, updates, resp)
	return resp, err
}

func (c *ConfigItemStatusClient) List(opts *ListOpts) (*ConfigItemStatusCollection, error) {
	resp := &ConfigItemStatusCollection{}
	err := c.rancherClient.doList(CONFIG_ITEM_STATUS_TYPE, opts, resp)
	return resp, err
}

func (c *ConfigItemStatusClient) ById(id string) (*ConfigItemStatus, error) {
	resp := &ConfigItemStatus{}
	err := c.rancherClient.doById(CONFIG_ITEM_STATUS_TYPE, id, resp)
	if apiError, ok := err.(*ApiError); ok {
		if apiError.StatusCode == 404 {
			return nil, nil
		}
	}
	return resp, err
}

func (c *ConfigItemStatusClient) Delete(container *ConfigItemStatus) error {
	return c.rancherClient.doResourceDelete(CONFIG_ITEM_STATUS_TYPE, &container.Resource)
}
