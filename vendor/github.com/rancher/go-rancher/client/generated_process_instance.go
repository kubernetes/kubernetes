package client

const (
	PROCESS_INSTANCE_TYPE = "processInstance"
)

type ProcessInstance struct {
	Resource

	Data map[string]interface{} `json:"data,omitempty" yaml:"data,omitempty"`

	EndTime string `json:"endTime,omitempty" yaml:"end_time,omitempty"`

	ExitReason string `json:"exitReason,omitempty" yaml:"exit_reason,omitempty"`

	Phase string `json:"phase,omitempty" yaml:"phase,omitempty"`

	Priority int64 `json:"priority,omitempty" yaml:"priority,omitempty"`

	ProcessName string `json:"processName,omitempty" yaml:"process_name,omitempty"`

	ResourceId string `json:"resourceId,omitempty" yaml:"resource_id,omitempty"`

	ResourceType string `json:"resourceType,omitempty" yaml:"resource_type,omitempty"`

	Result string `json:"result,omitempty" yaml:"result,omitempty"`

	RunningProcessServerId string `json:"runningProcessServerId,omitempty" yaml:"running_process_server_id,omitempty"`

	StartProcessServerId string `json:"startProcessServerId,omitempty" yaml:"start_process_server_id,omitempty"`

	StartTime string `json:"startTime,omitempty" yaml:"start_time,omitempty"`
}

type ProcessInstanceCollection struct {
	Collection
	Data []ProcessInstance `json:"data,omitempty"`
}

type ProcessInstanceClient struct {
	rancherClient *RancherClient
}

type ProcessInstanceOperations interface {
	List(opts *ListOpts) (*ProcessInstanceCollection, error)
	Create(opts *ProcessInstance) (*ProcessInstance, error)
	Update(existing *ProcessInstance, updates interface{}) (*ProcessInstance, error)
	ById(id string) (*ProcessInstance, error)
	Delete(container *ProcessInstance) error
}

func newProcessInstanceClient(rancherClient *RancherClient) *ProcessInstanceClient {
	return &ProcessInstanceClient{
		rancherClient: rancherClient,
	}
}

func (c *ProcessInstanceClient) Create(container *ProcessInstance) (*ProcessInstance, error) {
	resp := &ProcessInstance{}
	err := c.rancherClient.doCreate(PROCESS_INSTANCE_TYPE, container, resp)
	return resp, err
}

func (c *ProcessInstanceClient) Update(existing *ProcessInstance, updates interface{}) (*ProcessInstance, error) {
	resp := &ProcessInstance{}
	err := c.rancherClient.doUpdate(PROCESS_INSTANCE_TYPE, &existing.Resource, updates, resp)
	return resp, err
}

func (c *ProcessInstanceClient) List(opts *ListOpts) (*ProcessInstanceCollection, error) {
	resp := &ProcessInstanceCollection{}
	err := c.rancherClient.doList(PROCESS_INSTANCE_TYPE, opts, resp)
	return resp, err
}

func (c *ProcessInstanceClient) ById(id string) (*ProcessInstance, error) {
	resp := &ProcessInstance{}
	err := c.rancherClient.doById(PROCESS_INSTANCE_TYPE, id, resp)
	if apiError, ok := err.(*ApiError); ok {
		if apiError.StatusCode == 404 {
			return nil, nil
		}
	}
	return resp, err
}

func (c *ProcessInstanceClient) Delete(container *ProcessInstance) error {
	return c.rancherClient.doResourceDelete(PROCESS_INSTANCE_TYPE, &container.Resource)
}
