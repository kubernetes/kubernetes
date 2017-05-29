package client

const (
	SUBSCRIBE_TYPE = "subscribe"
)

type Subscribe struct {
	Resource

	AgentId string `json:"agentId,omitempty" yaml:"agent_id,omitempty"`

	EventNames []string `json:"eventNames,omitempty" yaml:"event_names,omitempty"`
}

type SubscribeCollection struct {
	Collection
	Data []Subscribe `json:"data,omitempty"`
}

type SubscribeClient struct {
	rancherClient *RancherClient
}

type SubscribeOperations interface {
	List(opts *ListOpts) (*SubscribeCollection, error)
	Create(opts *Subscribe) (*Subscribe, error)
	Update(existing *Subscribe, updates interface{}) (*Subscribe, error)
	ById(id string) (*Subscribe, error)
	Delete(container *Subscribe) error
}

func newSubscribeClient(rancherClient *RancherClient) *SubscribeClient {
	return &SubscribeClient{
		rancherClient: rancherClient,
	}
}

func (c *SubscribeClient) Create(container *Subscribe) (*Subscribe, error) {
	resp := &Subscribe{}
	err := c.rancherClient.doCreate(SUBSCRIBE_TYPE, container, resp)
	return resp, err
}

func (c *SubscribeClient) Update(existing *Subscribe, updates interface{}) (*Subscribe, error) {
	resp := &Subscribe{}
	err := c.rancherClient.doUpdate(SUBSCRIBE_TYPE, &existing.Resource, updates, resp)
	return resp, err
}

func (c *SubscribeClient) List(opts *ListOpts) (*SubscribeCollection, error) {
	resp := &SubscribeCollection{}
	err := c.rancherClient.doList(SUBSCRIBE_TYPE, opts, resp)
	return resp, err
}

func (c *SubscribeClient) ById(id string) (*Subscribe, error) {
	resp := &Subscribe{}
	err := c.rancherClient.doById(SUBSCRIBE_TYPE, id, resp)
	if apiError, ok := err.(*ApiError); ok {
		if apiError.StatusCode == 404 {
			return nil, nil
		}
	}
	return resp, err
}

func (c *SubscribeClient) Delete(container *Subscribe) error {
	return c.rancherClient.doResourceDelete(SUBSCRIBE_TYPE, &container.Resource)
}
