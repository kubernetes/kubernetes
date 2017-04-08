package client

const (
	LABEL_TYPE = "label"
)

type Label struct {
	Resource

	AccountId string `json:"accountId,omitempty" yaml:"account_id,omitempty"`

	Created string `json:"created,omitempty" yaml:"created,omitempty"`

	Data map[string]interface{} `json:"data,omitempty" yaml:"data,omitempty"`

	Description string `json:"description,omitempty" yaml:"description,omitempty"`

	Key string `json:"key,omitempty" yaml:"key,omitempty"`

	Kind string `json:"kind,omitempty" yaml:"kind,omitempty"`

	Name string `json:"name,omitempty" yaml:"name,omitempty"`

	RemoveTime string `json:"removeTime,omitempty" yaml:"remove_time,omitempty"`

	Removed string `json:"removed,omitempty" yaml:"removed,omitempty"`

	State string `json:"state,omitempty" yaml:"state,omitempty"`

	Transitioning string `json:"transitioning,omitempty" yaml:"transitioning,omitempty"`

	TransitioningMessage string `json:"transitioningMessage,omitempty" yaml:"transitioning_message,omitempty"`

	TransitioningProgress int64 `json:"transitioningProgress,omitempty" yaml:"transitioning_progress,omitempty"`

	Uuid string `json:"uuid,omitempty" yaml:"uuid,omitempty"`

	Value string `json:"value,omitempty" yaml:"value,omitempty"`
}

type LabelCollection struct {
	Collection
	Data []Label `json:"data,omitempty"`
}

type LabelClient struct {
	rancherClient *RancherClient
}

type LabelOperations interface {
	List(opts *ListOpts) (*LabelCollection, error)
	Create(opts *Label) (*Label, error)
	Update(existing *Label, updates interface{}) (*Label, error)
	ById(id string) (*Label, error)
	Delete(container *Label) error

	ActionCreate(*Label) (*Label, error)

	ActionRemove(*Label) (*Label, error)
}

func newLabelClient(rancherClient *RancherClient) *LabelClient {
	return &LabelClient{
		rancherClient: rancherClient,
	}
}

func (c *LabelClient) Create(container *Label) (*Label, error) {
	resp := &Label{}
	err := c.rancherClient.doCreate(LABEL_TYPE, container, resp)
	return resp, err
}

func (c *LabelClient) Update(existing *Label, updates interface{}) (*Label, error) {
	resp := &Label{}
	err := c.rancherClient.doUpdate(LABEL_TYPE, &existing.Resource, updates, resp)
	return resp, err
}

func (c *LabelClient) List(opts *ListOpts) (*LabelCollection, error) {
	resp := &LabelCollection{}
	err := c.rancherClient.doList(LABEL_TYPE, opts, resp)
	return resp, err
}

func (c *LabelClient) ById(id string) (*Label, error) {
	resp := &Label{}
	err := c.rancherClient.doById(LABEL_TYPE, id, resp)
	if apiError, ok := err.(*ApiError); ok {
		if apiError.StatusCode == 404 {
			return nil, nil
		}
	}
	return resp, err
}

func (c *LabelClient) Delete(container *Label) error {
	return c.rancherClient.doResourceDelete(LABEL_TYPE, &container.Resource)
}

func (c *LabelClient) ActionCreate(resource *Label) (*Label, error) {

	resp := &Label{}

	err := c.rancherClient.doAction(LABEL_TYPE, "create", &resource.Resource, nil, resp)

	return resp, err
}

func (c *LabelClient) ActionRemove(resource *Label) (*Label, error) {

	resp := &Label{}

	err := c.rancherClient.doAction(LABEL_TYPE, "remove", &resource.Resource, nil, resp)

	return resp, err
}
