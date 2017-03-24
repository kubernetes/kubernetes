package containerd

import "time"

type EventType int

func (t EventType) String() string {
	switch t {
	case ExitEvent:
		return "exit"
	case PausedEvent:
		return "paused"
	case CreateEvent:
		return "create"
	case StartEvent:
		return "start"
	case OOMEvent:
		return "oom"
	case ExecAddEvent:
		return "execAdd"
	}
	return "unknown"
}

const (
	ExitEvent EventType = iota + 1
	PausedEvent
	CreateEvent
	StartEvent
	OOMEvent
	ExecAddEvent
)

type Event struct {
	Timestamp  time.Time
	Type       EventType
	Runtime    string
	ID         string
	Pid        uint32
	ExitStatus uint32
}
