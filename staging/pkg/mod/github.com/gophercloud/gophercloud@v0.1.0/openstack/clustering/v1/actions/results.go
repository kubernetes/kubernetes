package actions

import (
	"encoding/json"
	"time"

	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/pagination"
)

// Action represents a detailed Action.
type Action struct {
	Action       string                 `json:"action"`
	Cause        string                 `json:"cause"`
	CreatedAt    time.Time              `json:"-"`
	Data         map[string]interface{} `json:"data"`
	DependedBy   []string               `json:"depended_by"`
	DependsOn    []string               `json:"depends_on"`
	StartTime    float64                `json:"start_time"`
	EndTime      float64                `json:"end_time"`
	ID           string                 `json:"id"`
	Inputs       map[string]interface{} `json:"inputs"`
	Interval     int                    `json:"interval"`
	Name         string                 `json:"name"`
	Outputs      map[string]interface{} `json:"outputs"`
	Owner        string                 `json:"owner"`
	Project      string                 `json:"project"`
	Status       string                 `json:"status"`
	StatusReason string                 `json:"status_reason"`
	Target       string                 `json:"target"`
	Timeout      int                    `json:"timeout"`
	UpdatedAt    time.Time              `json:"-"`
	User         string                 `json:"user"`
}

func (r *Action) UnmarshalJSON(b []byte) error {
	type tmp Action
	var s struct {
		tmp
		CreatedAt string `json:"created_at"`
		UpdatedAt string `json:"updated_at"`
	}

	err := json.Unmarshal(b, &s)
	if err != nil {
		return err
	}

	*r = Action(s.tmp)

	if s.CreatedAt != "" {
		r.CreatedAt, err = time.Parse(time.RFC3339, s.CreatedAt)
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

// Extract interprets any commonResult-based result as an Action.
func (r commonResult) Extract() (*Action, error) {
	var s struct {
		Action *Action `json:"action"`
	}
	err := r.ExtractInto(&s)
	return s.Action, err
}

// GetResult is the response of a Get operations. Call its Extract method to
// interpret it as an Action.
type GetResult struct {
	commonResult
}

// ActionPage contains a single page of all actions from a List call.
type ActionPage struct {
	pagination.LinkedPageBase
}

// IsEmpty determines if a ActionPage contains any results.
func (r ActionPage) IsEmpty() (bool, error) {
	actions, err := ExtractActions(r)
	return len(actions) == 0, err
}

// ExtractActions returns a slice of Actions from the List operation.
func ExtractActions(r pagination.Page) ([]Action, error) {
	var s struct {
		Actions []Action `json:"actions"`
	}
	err := (r.(ActionPage)).ExtractInto(&s)
	return s.Actions, err
}
