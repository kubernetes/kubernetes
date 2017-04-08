package client

const (
	PROCESS_DEFINITION_TYPE = "processDefinition"
)

type ProcessDefinition struct {
	Resource

	ExtensionBased bool `json:"extensionBased,omitempty" yaml:"extension_based,omitempty"`

	Name string `json:"name,omitempty" yaml:"name,omitempty"`

	PostProcessListeners interface{} `json:"postProcessListeners,omitempty" yaml:"post_process_listeners,omitempty"`

	PreProcessListeners interface{} `json:"preProcessListeners,omitempty" yaml:"pre_process_listeners,omitempty"`

	ProcessHandlers interface{} `json:"processHandlers,omitempty" yaml:"process_handlers,omitempty"`

	ResourceType string `json:"resourceType,omitempty" yaml:"resource_type,omitempty"`

	StateTransitions []interface{} `json:"stateTransitions,omitempty" yaml:"state_transitions,omitempty"`
}

type ProcessDefinitionCollection struct {
	Collection
	Data []ProcessDefinition `json:"data,omitempty"`
}

type ProcessDefinitionClient struct {
	rancherClient *RancherClient
}

type ProcessDefinitionOperations interface {
	List(opts *ListOpts) (*ProcessDefinitionCollection, error)
	Create(opts *ProcessDefinition) (*ProcessDefinition, error)
	Update(existing *ProcessDefinition, updates interface{}) (*ProcessDefinition, error)
	ById(id string) (*ProcessDefinition, error)
	Delete(container *ProcessDefinition) error
}

func newProcessDefinitionClient(rancherClient *RancherClient) *ProcessDefinitionClient {
	return &ProcessDefinitionClient{
		rancherClient: rancherClient,
	}
}

func (c *ProcessDefinitionClient) Create(container *ProcessDefinition) (*ProcessDefinition, error) {
	resp := &ProcessDefinition{}
	err := c.rancherClient.doCreate(PROCESS_DEFINITION_TYPE, container, resp)
	return resp, err
}

func (c *ProcessDefinitionClient) Update(existing *ProcessDefinition, updates interface{}) (*ProcessDefinition, error) {
	resp := &ProcessDefinition{}
	err := c.rancherClient.doUpdate(PROCESS_DEFINITION_TYPE, &existing.Resource, updates, resp)
	return resp, err
}

func (c *ProcessDefinitionClient) List(opts *ListOpts) (*ProcessDefinitionCollection, error) {
	resp := &ProcessDefinitionCollection{}
	err := c.rancherClient.doList(PROCESS_DEFINITION_TYPE, opts, resp)
	return resp, err
}

func (c *ProcessDefinitionClient) ById(id string) (*ProcessDefinition, error) {
	resp := &ProcessDefinition{}
	err := c.rancherClient.doById(PROCESS_DEFINITION_TYPE, id, resp)
	if apiError, ok := err.(*ApiError); ok {
		if apiError.StatusCode == 404 {
			return nil, nil
		}
	}
	return resp, err
}

func (c *ProcessDefinitionClient) Delete(container *ProcessDefinition) error {
	return c.rancherClient.doResourceDelete(PROCESS_DEFINITION_TYPE, &container.Resource)
}
