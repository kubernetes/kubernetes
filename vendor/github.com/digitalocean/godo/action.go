package godo

import "fmt"

const (
	actionsBasePath = "v2/actions"

	// ActionInProgress is an in progress action status
	ActionInProgress = "in-progress"

	//ActionCompleted is a completed action status
	ActionCompleted = "completed"
)

// ActionsService handles communction with action related methods of the
// DigitalOcean API: https://developers.digitalocean.com/documentation/v2#actions
type ActionsService interface {
	List(*ListOptions) ([]Action, *Response, error)
	Get(int) (*Action, *Response, error)
}

// ActionsServiceOp handles communition with the image action related methods of the
// DigitalOcean API.
type ActionsServiceOp struct {
	client *Client
}

var _ ActionsService = &ActionsServiceOp{}

type actionsRoot struct {
	Actions []Action `json:"actions"`
	Links   *Links   `json:"links"`
}

type actionRoot struct {
	Event Action `json:"action"`
}

// Action represents a DigitalOcean Action
type Action struct {
	ID           int        `json:"id"`
	Status       string     `json:"status"`
	Type         string     `json:"type"`
	StartedAt    *Timestamp `json:"started_at"`
	CompletedAt  *Timestamp `json:"completed_at"`
	ResourceID   int        `json:"resource_id"`
	ResourceType string     `json:"resource_type"`
	Region       *Region    `json:"region,omitempty"`
	RegionSlug   string     `json:"region_slug,omitempty"`
}

// List all actions
func (s *ActionsServiceOp) List(opt *ListOptions) ([]Action, *Response, error) {
	path := actionsBasePath
	path, err := addOptions(path, opt)
	if err != nil {
		return nil, nil, err
	}

	req, err := s.client.NewRequest("GET", path, nil)
	if err != nil {
		return nil, nil, err
	}

	root := new(actionsRoot)
	resp, err := s.client.Do(req, root)
	if err != nil {
		return nil, resp, err
	}
	if l := root.Links; l != nil {
		resp.Links = l
	}

	return root.Actions, resp, err
}

// Get an action by ID.
func (s *ActionsServiceOp) Get(id int) (*Action, *Response, error) {
	if id < 1 {
		return nil, nil, NewArgError("id", "cannot be less than 1")
	}

	path := fmt.Sprintf("%s/%d", actionsBasePath, id)
	req, err := s.client.NewRequest("GET", path, nil)
	if err != nil {
		return nil, nil, err
	}

	root := new(actionRoot)
	resp, err := s.client.Do(req, root)
	if err != nil {
		return nil, resp, err
	}

	return &root.Event, resp, err
}

func (a Action) String() string {
	return Stringify(a)
}
