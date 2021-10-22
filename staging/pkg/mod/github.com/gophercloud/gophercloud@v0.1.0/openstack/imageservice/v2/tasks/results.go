package tasks

import (
	"time"

	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/pagination"
)

type commonResult struct {
	gophercloud.Result
}

// GetResult represents the result of a Get operation. Call its Extract
// method to interpret it as a Task.
type GetResult struct {
	commonResult
}

// CreateResult represents the result of a Create operation. Call its Extract
// method to interpret it as a Task.
type CreateResult struct {
	commonResult
}

// Task represents a single task of the OpenStack Image service.
type Task struct {
	// ID is a unique identifier of the task.
	ID string `json:"id"`

	// Type represents the type of the task.
	Type string `json:"type"`

	// Status represents current status of the task.
	// You can use the TaskStatus custom type to unmarshal raw JSON response into
	// the pre-defined valid task status.
	Status string `json:"status"`

	// Input represents different parameters for the task.
	Input map[string]interface{} `json:"input"`

	// Result represents task result details.
	Result map[string]interface{} `json:"result"`

	// Owner is a unique identifier of the task owner.
	Owner string `json:"owner"`

	// Message represents human-readable message that is usually populated
	// on task failure.
	Message string `json:"message"`

	// ExpiresAt contains the timestamp of when the task will become a subject of
	// removal.
	ExpiresAt time.Time `json:"expires_at"`

	// CreatedAt contains the task creation timestamp.
	CreatedAt time.Time `json:"created_at"`

	// UpdatedAt contains the latest timestamp of when the task was updated.
	UpdatedAt time.Time `json:"updated_at"`

	// Self contains URI for the task.
	Self string `json:"self"`

	// Schema the path to the JSON-schema that represent the task.
	Schema string `json:"schema"`
}

// Extract interprets any commonResult as a Task.
func (r commonResult) Extract() (*Task, error) {
	var s *Task
	err := r.ExtractInto(&s)
	return s, err
}

// TaskPage represents the results of a List request.
type TaskPage struct {
	serviceURL string
	pagination.LinkedPageBase
}

// IsEmpty returns true if a TaskPage contains no Tasks results.
func (r TaskPage) IsEmpty() (bool, error) {
	tasks, err := ExtractTasks(r)
	return len(tasks) == 0, err
}

// NextPageURL uses the response's embedded link reference to navigate to
// the next page of results.
func (r TaskPage) NextPageURL() (string, error) {
	var s struct {
		Next string `json:"next"`
	}
	err := r.ExtractInto(&s)
	if err != nil {
		return "", err
	}

	if s.Next == "" {
		return "", nil
	}

	return nextPageURL(r.serviceURL, s.Next)
}

// ExtractTasks interprets the results of a single page from a List() call,
// producing a slice of Task entities.
func ExtractTasks(r pagination.Page) ([]Task, error) {
	var s struct {
		Tasks []Task `json:"tasks"`
	}
	err := (r.(TaskPage)).ExtractInto(&s)
	return s.Tasks, err
}
