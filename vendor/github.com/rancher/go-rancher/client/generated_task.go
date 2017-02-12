package client

const (
	TASK_TYPE = "task"
)

type Task struct {
	Resource

	Name string `json:"name,omitempty" yaml:"name,omitempty"`
}

type TaskCollection struct {
	Collection
	Data []Task `json:"data,omitempty"`
}

type TaskClient struct {
	rancherClient *RancherClient
}

type TaskOperations interface {
	List(opts *ListOpts) (*TaskCollection, error)
	Create(opts *Task) (*Task, error)
	Update(existing *Task, updates interface{}) (*Task, error)
	ById(id string) (*Task, error)
	Delete(container *Task) error

	ActionExecute(*Task) (*Task, error)
}

func newTaskClient(rancherClient *RancherClient) *TaskClient {
	return &TaskClient{
		rancherClient: rancherClient,
	}
}

func (c *TaskClient) Create(container *Task) (*Task, error) {
	resp := &Task{}
	err := c.rancherClient.doCreate(TASK_TYPE, container, resp)
	return resp, err
}

func (c *TaskClient) Update(existing *Task, updates interface{}) (*Task, error) {
	resp := &Task{}
	err := c.rancherClient.doUpdate(TASK_TYPE, &existing.Resource, updates, resp)
	return resp, err
}

func (c *TaskClient) List(opts *ListOpts) (*TaskCollection, error) {
	resp := &TaskCollection{}
	err := c.rancherClient.doList(TASK_TYPE, opts, resp)
	return resp, err
}

func (c *TaskClient) ById(id string) (*Task, error) {
	resp := &Task{}
	err := c.rancherClient.doById(TASK_TYPE, id, resp)
	if apiError, ok := err.(*ApiError); ok {
		if apiError.StatusCode == 404 {
			return nil, nil
		}
	}
	return resp, err
}

func (c *TaskClient) Delete(container *Task) error {
	return c.rancherClient.doResourceDelete(TASK_TYPE, &container.Resource)
}

func (c *TaskClient) ActionExecute(resource *Task) (*Task, error) {

	resp := &Task{}

	err := c.rancherClient.doAction(TASK_TYPE, "execute", &resource.Resource, nil, resp)

	return resp, err
}
