package events

import (
	"time"

	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/pagination"
)

// Event represents a detailed Event.
type Event struct {
	Action       string                 `json:"action"`
	Cluster      string                 `json:"cluster"`
	ClusterID    string                 `json:"cluster_id"`
	ID           string                 `json:"id"`
	Level        string                 `json:"level"`
	Metadata     map[string]interface{} `json:"meta_data"`
	OID          string                 `json:"oid"`
	OName        string                 `json:"oname"`
	OType        string                 `json:"otype"`
	Project      string                 `json:"project"`
	Status       string                 `json:"status"`
	StatusReason string                 `json:"status_reason"`
	Timestamp    time.Time              `json:"timestamp"`
	User         string                 `json:"user"`
}

// commonResult is the response of a base result.
type commonResult struct {
	gophercloud.Result
}

// Extract interprets any commonResult-based result as an Event.
func (r commonResult) Extract() (*Event, error) {
	var s struct {
		Event *Event `json:"event"`
	}
	err := r.ExtractInto(&s)
	return s.Event, err
}

// GetResult is the response of a Get operations. Call its Extract method to
// interpret it as an Event.
type GetResult struct {
	commonResult
}

// EventPage contains a single page of all events from a List call.
type EventPage struct {
	pagination.LinkedPageBase
}

// IsEmpty determines if a EventPage contains any results.
func (r EventPage) IsEmpty() (bool, error) {
	events, err := ExtractEvents(r)
	return len(events) == 0, err
}

// ExtractEvents returns a slice of Events from the List operation.
func ExtractEvents(r pagination.Page) ([]Event, error) {
	var s struct {
		Events []Event `json:"events"`
	}
	err := (r.(EventPage)).ExtractInto(&s)
	return s.Events, err
}
