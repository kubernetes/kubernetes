package client

const (
	PULL_TASK_TYPE = "pullTask"
)

type PullTask struct {
	Resource

	AccountId string `json:"accountId,omitempty" yaml:"account_id,omitempty"`

	Created string `json:"created,omitempty" yaml:"created,omitempty"`

	Data map[string]interface{} `json:"data,omitempty" yaml:"data,omitempty"`

	Description string `json:"description,omitempty" yaml:"description,omitempty"`

	Image string `json:"image,omitempty" yaml:"image,omitempty"`

	Kind string `json:"kind,omitempty" yaml:"kind,omitempty"`

	Labels map[string]interface{} `json:"labels,omitempty" yaml:"labels,omitempty"`

	Mode string `json:"mode,omitempty" yaml:"mode,omitempty"`

	Name string `json:"name,omitempty" yaml:"name,omitempty"`

	RemoveTime string `json:"removeTime,omitempty" yaml:"remove_time,omitempty"`

	Removed string `json:"removed,omitempty" yaml:"removed,omitempty"`

	State string `json:"state,omitempty" yaml:"state,omitempty"`

	Status map[string]interface{} `json:"status,omitempty" yaml:"status,omitempty"`

	Transitioning string `json:"transitioning,omitempty" yaml:"transitioning,omitempty"`

	TransitioningMessage string `json:"transitioningMessage,omitempty" yaml:"transitioning_message,omitempty"`

	TransitioningProgress int64 `json:"transitioningProgress,omitempty" yaml:"transitioning_progress,omitempty"`

	Uuid string `json:"uuid,omitempty" yaml:"uuid,omitempty"`
}

type PullTaskCollection struct {
	Collection
	Data []PullTask `json:"data,omitempty"`
}

type PullTaskClient struct {
	rancherClient *RancherClient
}

type PullTaskOperations interface {
	List(opts *ListOpts) (*PullTaskCollection, error)
	Create(opts *PullTask) (*PullTask, error)
	Update(existing *PullTask, updates interface{}) (*PullTask, error)
	ById(id string) (*PullTask, error)
	Delete(container *PullTask) error
}

func newPullTaskClient(rancherClient *RancherClient) *PullTaskClient {
	return &PullTaskClient{
		rancherClient: rancherClient,
	}
}

func (c *PullTaskClient) Create(container *PullTask) (*PullTask, error) {
	resp := &PullTask{}
	err := c.rancherClient.doCreate(PULL_TASK_TYPE, container, resp)
	return resp, err
}

func (c *PullTaskClient) Update(existing *PullTask, updates interface{}) (*PullTask, error) {
	resp := &PullTask{}
	err := c.rancherClient.doUpdate(PULL_TASK_TYPE, &existing.Resource, updates, resp)
	return resp, err
}

func (c *PullTaskClient) List(opts *ListOpts) (*PullTaskCollection, error) {
	resp := &PullTaskCollection{}
	err := c.rancherClient.doList(PULL_TASK_TYPE, opts, resp)
	return resp, err
}

func (c *PullTaskClient) ById(id string) (*PullTask, error) {
	resp := &PullTask{}
	err := c.rancherClient.doById(PULL_TASK_TYPE, id, resp)
	if apiError, ok := err.(*ApiError); ok {
		if apiError.StatusCode == 404 {
			return nil, nil
		}
	}
	return resp, err
}

func (c *PullTaskClient) Delete(container *PullTask) error {
	return c.rancherClient.doResourceDelete(PULL_TASK_TYPE, &container.Resource)
}
