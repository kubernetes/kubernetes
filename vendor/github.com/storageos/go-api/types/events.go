package types

import "time"

// EventType describes the type of event
type EventType string

// EventTypes are added to events to assist with type assertions
const (
	RequestType   EventType = "request"
	ResponseType            = "response"
	HeartbeatType           = "heartbeat"
	BackupType              = "backup"
)

// Event describes the fields that all events should implement.  Event is
// intended to be inherherited in more specific Event types.
type Event struct {
	ID string `json:"id"`
	// Parent is used to specify parent event
	Parent          string    `json:"parent"`
	EventType       EventType `json:"eventType"`
	Action          string    `json:"action"`
	Timestamp       int64     `json:"timestamp"`
	Status          string    `json:"status"`
	Message         string    `json:"message"`
	Log             []string  `json:"log"`
	ProgressPercent int       `json:"progressPercent"`
	CreatedBy       string    `json:"createdBy"`

	Target        string      `json:"target"`
	ActionPayload interface{} `json:"actionPayload"`

	// payload can be encoded into bytes as well
	ActionPayloadBytes []byte `json:"actionPayloadBts"`

	UpdatedAt time.Time `json:"updatedAt"`
	CreatedAt time.Time `json:"createdAt"`
	// retry related value
	Retry     bool      `json:"retry"`
	RetriedAt time.Time `json:"retriedAt"`
	Attempts  int       `json:"attempts"`

	// optional parameter
	Deadline time.Time `json:"deadline"`

	// optional events to dispatch
	Rollback     []*Request `json:"rollback"`
	RollbackDone bool       `json:"rollbackDone"`

	Subject string `json:"subject"` // or "queue"

	// controller ID which created this event
	OriginController string `json:"originController"`
}

// Request is the message structure used for sending request events
type Request struct {
	Event
}
