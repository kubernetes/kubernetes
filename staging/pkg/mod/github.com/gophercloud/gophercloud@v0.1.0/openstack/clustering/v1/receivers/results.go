package receivers

import (
	"encoding/json"
	"time"

	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/pagination"
)

// Receiver represent a detailed receiver
type Receiver struct {
	Action    string                 `json:"action"`
	Actor     map[string]interface{} `json:"actor"`
	Channel   map[string]interface{} `json:"channel"`
	ClusterID string                 `json:"cluster_id"`
	CreatedAt time.Time              `json:"-"`
	Domain    string                 `json:"domain"`
	ID        string                 `json:"id"`
	Name      string                 `json:"name"`
	Params    map[string]interface{} `json:"params"`
	Project   string                 `json:"project"`
	Type      string                 `json:"type"`
	UpdatedAt time.Time              `json:"-"`
	User      string                 `json:"user"`
}

func (r *Receiver) UnmarshalJSON(b []byte) error {
	type tmp Receiver
	var s struct {
		tmp
		CreatedAt string `json:"created_at"`
		UpdatedAt string `json:"updated_at"`
	}
	err := json.Unmarshal(b, &s)
	if err != nil {
		return err
	}
	*r = Receiver(s.tmp)

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

// Extract interprets any commonResult-based result as a Receiver.
func (r commonResult) Extract() (*Receiver, error) {
	var s struct {
		Receiver *Receiver `json:"receiver"`
	}
	err := r.ExtractInto(&s)
	return s.Receiver, err
}

// CreateResult is the result of a Create operation. Call its Extract method
// to interpret it as a Receiver.
type CreateResult struct {
	commonResult
}

// GetResult is the result for of a Get operation. Call its Extract method
// to interpret it as a Receiver.
type GetResult struct {
	commonResult
}

// UpdateResult is the result of a Update operation. Call its Extract method
// to interpret it as a Receiver.
type UpdateResult struct {
	commonResult
}

// DeleteResult is the result from a Delete operation. Call its ExtractErr
// method to determine if the call succeeded or failed.
type DeleteResult struct {
	gophercloud.ErrResult
}

// NotifyResult is the result from a Notify operation. Call its Extract
// method to determine if the call succeeded or failed.
type NotifyResult struct {
	commonResult
}

// ReceiverPage contains a single page of all nodes from a List operation.
type ReceiverPage struct {
	pagination.LinkedPageBase
}

// IsEmpty determines if a ReceiverPage contains any results.
func (page ReceiverPage) IsEmpty() (bool, error) {
	receivers, err := ExtractReceivers(page)
	return len(receivers) == 0, err
}

// ExtractReceivers returns a slice of Receivers from the List operation.
func ExtractReceivers(r pagination.Page) ([]Receiver, error) {
	var s struct {
		Receivers []Receiver `json:"receivers"`
	}
	err := (r.(ReceiverPage)).ExtractInto(&s)
	return s.Receivers, err
}

// Extract returns action for notify receivers
func (r NotifyResult) Extract() (string, error) {
	requestID := r.Header.Get("X-Openstack-Request-Id")
	return requestID, r.Err
}
