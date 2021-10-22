package extendedstatus

type PowerState int

type ServerExtendedStatusExt struct {
	TaskState  string     `json:"OS-EXT-STS:task_state"`
	VmState    string     `json:"OS-EXT-STS:vm_state"`
	PowerState PowerState `json:"OS-EXT-STS:power_state"`
}

const (
	NOSTATE = iota
	RUNNING
	_UNUSED1
	PAUSED
	SHUTDOWN
	_UNUSED2
	CRASHED
	SUSPENDED
)

func (r PowerState) String() string {
	switch r {
	case NOSTATE:
		return "NOSTATE"
	case RUNNING:
		return "RUNNING"
	case PAUSED:
		return "PAUSED"
	case SHUTDOWN:
		return "SHUTDOWN"
	case CRASHED:
		return "CRASHED"
	case SUSPENDED:
		return "SUSPENDED"
	case _UNUSED1, _UNUSED2:
		return "_UNUSED"
	default:
		return "N/A"
	}
}
