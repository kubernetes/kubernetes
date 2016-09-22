package client

const (
	CONFIG_ITEM_TYPE = "configItem"
)

type ConfigItem struct {
	Resource

	Name string `json:"name,omitempty" yaml:"name,omitempty"`

	SourceVersion string `json:"sourceVersion,omitempty" yaml:"source_version,omitempty"`
}

type ConfigItemCollection struct {
	Collection
	Data []ConfigItem `json:"data,omitempty"`
}

type ConfigItemClient struct {
	rancherClient *RancherClient
}

type ConfigItemOperations interface {
	List(opts *ListOpts) (*ConfigItemCollection, error)
	Create(opts *ConfigItem) (*ConfigItem, error)
	Update(existing *ConfigItem, updates interface{}) (*ConfigItem, error)
	ById(id string) (*ConfigItem, error)
	Delete(container *ConfigItem) error
}

func newConfigItemClient(rancherClient *RancherClient) *ConfigItemClient {
	return &ConfigItemClient{
		rancherClient: rancherClient,
	}
}

func (c *ConfigItemClient) Create(container *ConfigItem) (*ConfigItem, error) {
	resp := &ConfigItem{}
	err := c.rancherClient.doCreate(CONFIG_ITEM_TYPE, container, resp)
	return resp, err
}

func (c *ConfigItemClient) Update(existing *ConfigItem, updates interface{}) (*ConfigItem, error) {
	resp := &ConfigItem{}
	err := c.rancherClient.doUpdate(CONFIG_ITEM_TYPE, &existing.Resource, updates, resp)
	return resp, err
}

func (c *ConfigItemClient) List(opts *ListOpts) (*ConfigItemCollection, error) {
	resp := &ConfigItemCollection{}
	err := c.rancherClient.doList(CONFIG_ITEM_TYPE, opts, resp)
	return resp, err
}

func (c *ConfigItemClient) ById(id string) (*ConfigItem, error) {
	resp := &ConfigItem{}
	err := c.rancherClient.doById(CONFIG_ITEM_TYPE, id, resp)
	if apiError, ok := err.(*ApiError); ok {
		if apiError.StatusCode == 404 {
			return nil, nil
		}
	}
	return resp, err
}

func (c *ConfigItemClient) Delete(container *ConfigItem) error {
	return c.rancherClient.doResourceDelete(CONFIG_ITEM_TYPE, &container.Resource)
}
