package client

const (
	PROJECT_MEMBER_TYPE = "projectMember"
)

type ProjectMember struct {
	Resource

	Created string `json:"created,omitempty" yaml:"created,omitempty"`

	Data map[string]interface{} `json:"data,omitempty" yaml:"data,omitempty"`

	Description string `json:"description,omitempty" yaml:"description,omitempty"`

	ExternalId string `json:"externalId,omitempty" yaml:"external_id,omitempty"`

	ExternalIdType string `json:"externalIdType,omitempty" yaml:"external_id_type,omitempty"`

	Kind string `json:"kind,omitempty" yaml:"kind,omitempty"`

	Name string `json:"name,omitempty" yaml:"name,omitempty"`

	ProjectId string `json:"projectId,omitempty" yaml:"project_id,omitempty"`

	RemoveTime string `json:"removeTime,omitempty" yaml:"remove_time,omitempty"`

	Removed string `json:"removed,omitempty" yaml:"removed,omitempty"`

	Role string `json:"role,omitempty" yaml:"role,omitempty"`

	State string `json:"state,omitempty" yaml:"state,omitempty"`

	Transitioning string `json:"transitioning,omitempty" yaml:"transitioning,omitempty"`

	TransitioningMessage string `json:"transitioningMessage,omitempty" yaml:"transitioning_message,omitempty"`

	TransitioningProgress int64 `json:"transitioningProgress,omitempty" yaml:"transitioning_progress,omitempty"`

	Uuid string `json:"uuid,omitempty" yaml:"uuid,omitempty"`
}

type ProjectMemberCollection struct {
	Collection
	Data []ProjectMember `json:"data,omitempty"`
}

type ProjectMemberClient struct {
	rancherClient *RancherClient
}

type ProjectMemberOperations interface {
	List(opts *ListOpts) (*ProjectMemberCollection, error)
	Create(opts *ProjectMember) (*ProjectMember, error)
	Update(existing *ProjectMember, updates interface{}) (*ProjectMember, error)
	ById(id string) (*ProjectMember, error)
	Delete(container *ProjectMember) error

	ActionActivate(*ProjectMember) (*ProjectMember, error)

	ActionCreate(*ProjectMember) (*ProjectMember, error)

	ActionDeactivate(*ProjectMember) (*ProjectMember, error)

	ActionPurge(*ProjectMember) (*ProjectMember, error)

	ActionRemove(*ProjectMember) (*ProjectMember, error)

	ActionRestore(*ProjectMember) (*ProjectMember, error)

	ActionUpdate(*ProjectMember) (*ProjectMember, error)
}

func newProjectMemberClient(rancherClient *RancherClient) *ProjectMemberClient {
	return &ProjectMemberClient{
		rancherClient: rancherClient,
	}
}

func (c *ProjectMemberClient) Create(container *ProjectMember) (*ProjectMember, error) {
	resp := &ProjectMember{}
	err := c.rancherClient.doCreate(PROJECT_MEMBER_TYPE, container, resp)
	return resp, err
}

func (c *ProjectMemberClient) Update(existing *ProjectMember, updates interface{}) (*ProjectMember, error) {
	resp := &ProjectMember{}
	err := c.rancherClient.doUpdate(PROJECT_MEMBER_TYPE, &existing.Resource, updates, resp)
	return resp, err
}

func (c *ProjectMemberClient) List(opts *ListOpts) (*ProjectMemberCollection, error) {
	resp := &ProjectMemberCollection{}
	err := c.rancherClient.doList(PROJECT_MEMBER_TYPE, opts, resp)
	return resp, err
}

func (c *ProjectMemberClient) ById(id string) (*ProjectMember, error) {
	resp := &ProjectMember{}
	err := c.rancherClient.doById(PROJECT_MEMBER_TYPE, id, resp)
	if apiError, ok := err.(*ApiError); ok {
		if apiError.StatusCode == 404 {
			return nil, nil
		}
	}
	return resp, err
}

func (c *ProjectMemberClient) Delete(container *ProjectMember) error {
	return c.rancherClient.doResourceDelete(PROJECT_MEMBER_TYPE, &container.Resource)
}

func (c *ProjectMemberClient) ActionActivate(resource *ProjectMember) (*ProjectMember, error) {

	resp := &ProjectMember{}

	err := c.rancherClient.doAction(PROJECT_MEMBER_TYPE, "activate", &resource.Resource, nil, resp)

	return resp, err
}

func (c *ProjectMemberClient) ActionCreate(resource *ProjectMember) (*ProjectMember, error) {

	resp := &ProjectMember{}

	err := c.rancherClient.doAction(PROJECT_MEMBER_TYPE, "create", &resource.Resource, nil, resp)

	return resp, err
}

func (c *ProjectMemberClient) ActionDeactivate(resource *ProjectMember) (*ProjectMember, error) {

	resp := &ProjectMember{}

	err := c.rancherClient.doAction(PROJECT_MEMBER_TYPE, "deactivate", &resource.Resource, nil, resp)

	return resp, err
}

func (c *ProjectMemberClient) ActionPurge(resource *ProjectMember) (*ProjectMember, error) {

	resp := &ProjectMember{}

	err := c.rancherClient.doAction(PROJECT_MEMBER_TYPE, "purge", &resource.Resource, nil, resp)

	return resp, err
}

func (c *ProjectMemberClient) ActionRemove(resource *ProjectMember) (*ProjectMember, error) {

	resp := &ProjectMember{}

	err := c.rancherClient.doAction(PROJECT_MEMBER_TYPE, "remove", &resource.Resource, nil, resp)

	return resp, err
}

func (c *ProjectMemberClient) ActionRestore(resource *ProjectMember) (*ProjectMember, error) {

	resp := &ProjectMember{}

	err := c.rancherClient.doAction(PROJECT_MEMBER_TYPE, "restore", &resource.Resource, nil, resp)

	return resp, err
}

func (c *ProjectMemberClient) ActionUpdate(resource *ProjectMember) (*ProjectMember, error) {

	resp := &ProjectMember{}

	err := c.rancherClient.doAction(PROJECT_MEMBER_TYPE, "update", &resource.Resource, nil, resp)

	return resp, err
}
