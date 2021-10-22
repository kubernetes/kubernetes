package nodes

import (
	"encoding/json"
	"time"

	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/pagination"
)

// Node represents an OpenStack clustering node.
type Node struct {
	ClusterID    string                 `json:"cluster_id"`
	CreatedAt    time.Time              `json:"-"`
	Data         map[string]interface{} `json:"data"`
	Dependents   map[string]interface{} `json:"dependents"`
	Domain       string                 `json:"domain"`
	ID           string                 `json:"id"`
	Index        int                    `json:"index"`
	InitAt       time.Time              `json:"-"`
	Metadata     map[string]interface{} `json:"metadata"`
	Name         string                 `json:"name"`
	PhysicalID   string                 `json:"physical_id"`
	ProfileID    string                 `json:"profile_id"`
	ProfileName  string                 `json:"profile_name"`
	Project      string                 `json:"project"`
	Role         string                 `json:"role"`
	Status       string                 `json:"status"`
	StatusReason string                 `json:"status_reason"`
	UpdatedAt    time.Time              `json:"-"`
	User         string                 `json:"user"`
}

func (r *Node) UnmarshalJSON(b []byte) error {
	type tmp Node
	var s struct {
		tmp
		CreatedAt string `json:"created_at"`
		InitAt    string `json:"init_at"`
		UpdatedAt string `json:"updated_at"`
	}

	err := json.Unmarshal(b, &s)
	if err != nil {
		return err
	}
	*r = Node(s.tmp)

	if s.CreatedAt != "" {
		r.CreatedAt, err = time.Parse(time.RFC3339, s.CreatedAt)
		if err != nil {
			return err
		}
	}

	if s.InitAt != "" {
		r.InitAt, err = time.Parse(time.RFC3339, s.InitAt)
		if err != nil {
			return err
		}
	}

	if s.UpdatedAt != "" {
		r.UpdatedAt, err = time.Parse(time.RFC3339, s.UpdatedAt)
		if err != nil {
			return err
		}
	}

	return nil
}

// commonResult is the response of a base result.
type commonResult struct {
	gophercloud.Result
}

// Extract interprets any commonResult-based result as a Node.
func (r commonResult) Extract() (*Node, error) {
	var s struct {
		Node *Node `json:"node"`
	}
	err := r.ExtractInto(&s)
	return s.Node, err
}

// CreateResult is the result of a Create operation. Call its Extract
// method to intepret it as a Node.
type CreateResult struct {
	commonResult
}

// GetResult is the result of a Get operation. Call its Extract method to
// interpret it as a Node.
type GetResult struct {
	commonResult
}

// DeleteResult is the result from a Delete operation. Call ExtractErr
// method to determine if the call succeeded or failed.
type DeleteResult struct {
	gophercloud.ErrResult
}

// UpdateResult is the result of an Update operation. Call its Extract method
// to interpet it as a Node.
type UpdateResult struct {
	commonResult
}

// NodePage contains a single page of all nodes from a List call.
type NodePage struct {
	pagination.LinkedPageBase
}

// IsEmpty determines if a NodePage contains any results.
func (page NodePage) IsEmpty() (bool, error) {
	nodes, err := ExtractNodes(page)
	return len(nodes) == 0, err
}

// ExtractNodes returns a slice of Nodes from the List operation.
func ExtractNodes(r pagination.Page) ([]Node, error) {
	var s struct {
		Nodes []Node `json:"nodes"`
	}
	err := (r.(NodePage)).ExtractInto(&s)
	return s.Nodes, err
}

// ActionResult is the response of Senlin actions. Call its Extract method to
// obtain the Action ID of the action.
type ActionResult struct {
	gophercloud.Result
}

// Extract interprets any Action result as an Action.
func (r ActionResult) Extract() (string, error) {
	var s struct {
		Action string `json:"action"`
	}
	err := r.ExtractInto(&s)
	return s.Action, err
}
