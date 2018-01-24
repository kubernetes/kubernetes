package githttp

import (
	"fmt"
	"net/http"
)

// An event (triggered on push/pull)
type Event struct {
	// One of tag/push/fetch
	Type EventType `json:"type"`

	////
	// Set for pushes and pulls
	////

	// SHA of commit
	Commit string `json:"commit"`

	// Path to bare repo
	Dir string

	////
	// Set for pushes or tagging
	////
	Tag    string `json:"tag,omitempty"`
	Last   string `json:"last,omitempty"`
	Branch string `json:"branch,omitempty"`

	// Error contains the error that happened (if any)
	// during this action/event
	Error error

	// Http stuff
	Request *http.Request
}

type EventType int

// Possible event types
const (
	TAG = iota + 1
	PUSH
	FETCH
	PUSH_FORCE
)

func (e EventType) String() string {
	switch e {
	case TAG:
		return "tag"
	case PUSH:
		return "push"
	case PUSH_FORCE:
		return "push-force"
	case FETCH:
		return "fetch"
	}
	return "unknown"
}

func (e EventType) MarshalJSON() ([]byte, error) {
	return []byte(fmt.Sprintf(`"%s"`, e)), nil
}

func (e EventType) UnmarshalJSON(data []byte) error {
	str := string(data[:])
	switch str {
	case "tag":
		e = TAG
	case "push":
		e = PUSH
	case "push-force":
		e = PUSH_FORCE
	case "fetch":
		e = FETCH
	default:
		return fmt.Errorf("'%s' is not a known git event type")
	}
	return nil
}
