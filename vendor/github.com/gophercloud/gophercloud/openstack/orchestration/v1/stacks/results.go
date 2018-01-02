package stacks

import (
	"encoding/json"
	"time"

	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/pagination"
)

// CreatedStack represents the object extracted from a Create operation.
type CreatedStack struct {
	ID    string             `json:"id"`
	Links []gophercloud.Link `json:"links"`
}

// CreateResult represents the result of a Create operation.
type CreateResult struct {
	gophercloud.Result
}

// Extract returns a pointer to a CreatedStack object and is called after a
// Create operation.
func (r CreateResult) Extract() (*CreatedStack, error) {
	var s struct {
		CreatedStack *CreatedStack `json:"stack"`
	}
	err := r.ExtractInto(&s)
	return s.CreatedStack, err
}

// AdoptResult represents the result of an Adopt operation. AdoptResult has the
// same form as CreateResult.
type AdoptResult struct {
	CreateResult
}

// StackPage is a pagination.Pager that is returned from a call to the List function.
type StackPage struct {
	pagination.SinglePageBase
}

// IsEmpty returns true if a ListResult contains no Stacks.
func (r StackPage) IsEmpty() (bool, error) {
	stacks, err := ExtractStacks(r)
	return len(stacks) == 0, err
}

// ListedStack represents an element in the slice extracted from a List operation.
type ListedStack struct {
	CreationTime time.Time          `json:"-"`
	Description  string             `json:"description"`
	ID           string             `json:"id"`
	Links        []gophercloud.Link `json:"links"`
	Name         string             `json:"stack_name"`
	Status       string             `json:"stack_status"`
	StatusReason string             `json:"stack_status_reason"`
	Tags         []string           `json:"tags"`
	UpdatedTime  time.Time          `json:"-"`
}

func (r *ListedStack) UnmarshalJSON(b []byte) error {
	type tmp ListedStack
	var s struct {
		tmp
		CreationTime gophercloud.JSONRFC3339NoZ `json:"creation_time"`
		UpdatedTime  gophercloud.JSONRFC3339NoZ `json:"updated_time"`
	}
	err := json.Unmarshal(b, &s)
	if err != nil {
		return err
	}
	*r = ListedStack(s.tmp)

	r.CreationTime = time.Time(s.CreationTime)
	r.UpdatedTime = time.Time(s.UpdatedTime)

	return nil
}

// ExtractStacks extracts and returns a slice of ListedStack. It is used while iterating
// over a stacks.List call.
func ExtractStacks(r pagination.Page) ([]ListedStack, error) {
	var s struct {
		ListedStacks []ListedStack `json:"stacks"`
	}
	err := (r.(StackPage)).ExtractInto(&s)
	return s.ListedStacks, err
}

// RetrievedStack represents the object extracted from a Get operation.
type RetrievedStack struct {
	Capabilities        []interface{}            `json:"capabilities"`
	CreationTime        time.Time                `json:"-"`
	Description         string                   `json:"description"`
	DisableRollback     bool                     `json:"disable_rollback"`
	ID                  string                   `json:"id"`
	Links               []gophercloud.Link       `json:"links"`
	NotificationTopics  []interface{}            `json:"notification_topics"`
	Outputs             []map[string]interface{} `json:"outputs"`
	Parameters          map[string]string        `json:"parameters"`
	Name                string                   `json:"stack_name"`
	Status              string                   `json:"stack_status"`
	StatusReason        string                   `json:"stack_status_reason"`
	Tags                []string                 `json:"tags"`
	TemplateDescription string                   `json:"template_description"`
	Timeout             int                      `json:"timeout_mins"`
	UpdatedTime         time.Time                `json:"-"`
}

func (r *RetrievedStack) UnmarshalJSON(b []byte) error {
	type tmp RetrievedStack
	var s struct {
		tmp
		CreationTime gophercloud.JSONRFC3339NoZ `json:"creation_time"`
		UpdatedTime  gophercloud.JSONRFC3339NoZ `json:"updated_time"`
	}
	err := json.Unmarshal(b, &s)
	if err != nil {
		return err
	}
	*r = RetrievedStack(s.tmp)

	r.CreationTime = time.Time(s.CreationTime)
	r.UpdatedTime = time.Time(s.UpdatedTime)

	return nil
}

// GetResult represents the result of a Get operation.
type GetResult struct {
	gophercloud.Result
}

// Extract returns a pointer to a RetrievedStack object and is called after a
// Get operation.
func (r GetResult) Extract() (*RetrievedStack, error) {
	var s struct {
		Stack *RetrievedStack `json:"stack"`
	}
	err := r.ExtractInto(&s)
	return s.Stack, err
}

// UpdateResult represents the result of a Update operation.
type UpdateResult struct {
	gophercloud.ErrResult
}

// DeleteResult represents the result of a Delete operation.
type DeleteResult struct {
	gophercloud.ErrResult
}

// PreviewedStack represents the result of a Preview operation.
type PreviewedStack struct {
	Capabilities        []interface{}      `json:"capabilities"`
	CreationTime        time.Time          `json:"-"`
	Description         string             `json:"description"`
	DisableRollback     bool               `json:"disable_rollback"`
	ID                  string             `json:"id"`
	Links               []gophercloud.Link `json:"links"`
	Name                string             `json:"stack_name"`
	NotificationTopics  []interface{}      `json:"notification_topics"`
	Parameters          map[string]string  `json:"parameters"`
	Resources           []interface{}      `json:"resources"`
	TemplateDescription string             `json:"template_description"`
	Timeout             int                `json:"timeout_mins"`
	UpdatedTime         time.Time          `json:"-"`
}

func (r *PreviewedStack) UnmarshalJSON(b []byte) error {
	type tmp PreviewedStack
	var s struct {
		tmp
		CreationTime gophercloud.JSONRFC3339NoZ `json:"creation_time"`
		UpdatedTime  gophercloud.JSONRFC3339NoZ `json:"updated_time"`
	}
	err := json.Unmarshal(b, &s)
	if err != nil {
		return err
	}
	*r = PreviewedStack(s.tmp)

	r.CreationTime = time.Time(s.CreationTime)
	r.UpdatedTime = time.Time(s.UpdatedTime)

	return nil
}

// PreviewResult represents the result of a Preview operation.
type PreviewResult struct {
	gophercloud.Result
}

// Extract returns a pointer to a PreviewedStack object and is called after a
// Preview operation.
func (r PreviewResult) Extract() (*PreviewedStack, error) {
	var s struct {
		PreviewedStack *PreviewedStack `json:"stack"`
	}
	err := r.ExtractInto(&s)
	return s.PreviewedStack, err
}

// AbandonedStack represents the result of an Abandon operation.
type AbandonedStack struct {
	Status             string                 `json:"status"`
	Name               string                 `json:"name"`
	Template           map[string]interface{} `json:"template"`
	Action             string                 `json:"action"`
	ID                 string                 `json:"id"`
	Resources          map[string]interface{} `json:"resources"`
	Files              map[string]string      `json:"files"`
	StackUserProjectID string                 `json:"stack_user_project_id"`
	ProjectID          string                 `json:"project_id"`
	Environment        map[string]interface{} `json:"environment"`
}

// AbandonResult represents the result of an Abandon operation.
type AbandonResult struct {
	gophercloud.Result
}

// Extract returns a pointer to an AbandonedStack object and is called after an
// Abandon operation.
func (r AbandonResult) Extract() (*AbandonedStack, error) {
	var s *AbandonedStack
	err := r.ExtractInto(&s)
	return s, err
}

// String converts an AbandonResult to a string. This is useful to when passing
// the result of an Abandon operation to an AdoptOpts AdoptStackData field.
func (r AbandonResult) String() (string, error) {
	out, err := json.Marshal(r)
	return string(out), err
}
