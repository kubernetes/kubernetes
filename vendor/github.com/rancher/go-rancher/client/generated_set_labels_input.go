package client

const (
	SET_LABELS_INPUT_TYPE = "setLabelsInput"
)

type SetLabelsInput struct {
	Resource

	Labels interface{} `json:"labels,omitempty" yaml:"labels,omitempty"`
}

type SetLabelsInputCollection struct {
	Collection
	Data []SetLabelsInput `json:"data,omitempty"`
}

type SetLabelsInputClient struct {
	rancherClient *RancherClient
}

type SetLabelsInputOperations interface {
	List(opts *ListOpts) (*SetLabelsInputCollection, error)
	Create(opts *SetLabelsInput) (*SetLabelsInput, error)
	Update(existing *SetLabelsInput, updates interface{}) (*SetLabelsInput, error)
	ById(id string) (*SetLabelsInput, error)
	Delete(container *SetLabelsInput) error
}

func newSetLabelsInputClient(rancherClient *RancherClient) *SetLabelsInputClient {
	return &SetLabelsInputClient{
		rancherClient: rancherClient,
	}
}

func (c *SetLabelsInputClient) Create(container *SetLabelsInput) (*SetLabelsInput, error) {
	resp := &SetLabelsInput{}
	err := c.rancherClient.doCreate(SET_LABELS_INPUT_TYPE, container, resp)
	return resp, err
}

func (c *SetLabelsInputClient) Update(existing *SetLabelsInput, updates interface{}) (*SetLabelsInput, error) {
	resp := &SetLabelsInput{}
	err := c.rancherClient.doUpdate(SET_LABELS_INPUT_TYPE, &existing.Resource, updates, resp)
	return resp, err
}

func (c *SetLabelsInputClient) List(opts *ListOpts) (*SetLabelsInputCollection, error) {
	resp := &SetLabelsInputCollection{}
	err := c.rancherClient.doList(SET_LABELS_INPUT_TYPE, opts, resp)
	return resp, err
}

func (c *SetLabelsInputClient) ById(id string) (*SetLabelsInput, error) {
	resp := &SetLabelsInput{}
	err := c.rancherClient.doById(SET_LABELS_INPUT_TYPE, id, resp)
	if apiError, ok := err.(*ApiError); ok {
		if apiError.StatusCode == 404 {
			return nil, nil
		}
	}
	return resp, err
}

func (c *SetLabelsInputClient) Delete(container *SetLabelsInput) error {
	return c.rancherClient.doResourceDelete(SET_LABELS_INPUT_TYPE, &container.Resource)
}
