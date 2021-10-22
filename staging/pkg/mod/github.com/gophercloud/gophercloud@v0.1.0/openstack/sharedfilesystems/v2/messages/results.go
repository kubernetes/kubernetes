package messages

import (
	"encoding/json"
	"time"

	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/pagination"
)

// Message contains all the information associated with an OpenStack
// Message.
type Message struct {
	// The message ID
	ID string `json:"id"`
	// The UUID of the project where the message was created
	ProjectID string `json:"project_id"`
	// The ID of the action during which the message was created
	ActionID string `json:"action_id"`
	// The ID of the message detail
	DetailID string `json:"detail_id"`
	// The message level
	MessageLevel string `json:"message_level"`
	// The UUID of the request during which the message was created
	RequestID string `json:"request_id"`
	// The UUID of the resource for which the message was created
	ResourceID string `json:"resource_id"`
	// The type of the resource for which the message was created
	ResourceType string `json:"resource_type"`
	// The message text
	UserMessage string `json:"user_message"`
	// The date and time stamp when the message was created
	CreatedAt time.Time `json:"-"`
	// The date and time stamp when the message will expire
	ExpiresAt time.Time `json:"-"`
}

func (r *Message) UnmarshalJSON(b []byte) error {
	type tmp Message
	var s struct {
		tmp
		CreatedAt gophercloud.JSONRFC3339MilliNoZ `json:"created_at"`
		ExpiresAt gophercloud.JSONRFC3339MilliNoZ `json:"expires_at"`
	}
	err := json.Unmarshal(b, &s)
	if err != nil {
		return err
	}
	*r = Message(s.tmp)

	r.CreatedAt = time.Time(s.CreatedAt)
	r.ExpiresAt = time.Time(s.ExpiresAt)

	return nil
}

type commonResult struct {
	gophercloud.Result
}

// MessagePage is a pagination.pager that is returned from a call to the List function.
type MessagePage struct {
	pagination.SinglePageBase
}

// IsEmpty returns true if a ListResult contains no Messages.
func (r MessagePage) IsEmpty() (bool, error) {
	messages, err := ExtractMessages(r)
	return len(messages) == 0, err
}

// ExtractMessages extracts and returns Messages. It is used while
// iterating over a messages.List call.
func ExtractMessages(r pagination.Page) ([]Message, error) {
	var s struct {
		Messages []Message `json:"messages"`
	}
	err := (r.(MessagePage)).ExtractInto(&s)
	return s.Messages, err
}

// Extract will get the Message object out of the commonResult object.
func (r commonResult) Extract() (*Message, error) {
	var s struct {
		Message *Message `json:"message"`
	}
	err := r.ExtractInto(&s)
	return s.Message, err
}

// DeleteResult contains the response body and error from a Delete request.
type DeleteResult struct {
	gophercloud.ErrResult
}

// GetResult contains the response body and error from a Get request.
type GetResult struct {
	commonResult
}
