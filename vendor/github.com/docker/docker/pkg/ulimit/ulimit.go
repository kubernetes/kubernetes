package ulimit

import (
	"fmt"
	"strconv"
	"strings"
)

// Human friendly version of Rlimit
type Ulimit struct {
	Name string
	Hard int64
	Soft int64
}

type Rlimit struct {
	Type int    `json:"type,omitempty"`
	Hard uint64 `json:"hard,omitempty"`
	Soft uint64 `json:"soft,omitempty"`
}

const (
	// magic numbers for making the syscall
	// some of these are defined in the syscall package, but not all.
	// Also since Windows client doesn't get access to the syscall package, need to
	//	define these here
	RLIMIT_AS         = 9
	RLIMIT_CORE       = 4
	RLIMIT_CPU        = 0
	RLIMIT_DATA       = 2
	RLIMIT_FSIZE      = 1
	RLIMIT_LOCKS      = 10
	RLIMIT_MEMLOCK    = 8
	RLIMIT_MSGQUEUE   = 12
	RLIMIT_NICE       = 13
	RLIMIT_NOFILE     = 7
	RLIMIT_NPROC      = 6
	RLIMIT_RSS        = 5
	RLIMIT_RTPRIO     = 14
	RLIMIT_RTTIME     = 15
	RLIMIT_SIGPENDING = 11
	RLIMIT_STACK      = 3
)

var ulimitNameMapping = map[string]int{
	//"as":         RLIMIT_AS, // Disbaled since this doesn't seem usable with the way Docker inits a container.
	"core":       RLIMIT_CORE,
	"cpu":        RLIMIT_CPU,
	"data":       RLIMIT_DATA,
	"fsize":      RLIMIT_FSIZE,
	"locks":      RLIMIT_LOCKS,
	"memlock":    RLIMIT_MEMLOCK,
	"msgqueue":   RLIMIT_MSGQUEUE,
	"nice":       RLIMIT_NICE,
	"nofile":     RLIMIT_NOFILE,
	"nproc":      RLIMIT_NPROC,
	"rss":        RLIMIT_RSS,
	"rtprio":     RLIMIT_RTPRIO,
	"rttime":     RLIMIT_RTTIME,
	"sigpending": RLIMIT_SIGPENDING,
	"stack":      RLIMIT_STACK,
}

func Parse(val string) (*Ulimit, error) {
	parts := strings.SplitN(val, "=", 2)
	if len(parts) != 2 {
		return nil, fmt.Errorf("invalid ulimit argument: %s", val)
	}

	if _, exists := ulimitNameMapping[parts[0]]; !exists {
		return nil, fmt.Errorf("invalid ulimit type: %s", parts[0])
	}

	limitVals := strings.SplitN(parts[1], ":", 2)
	if len(limitVals) > 2 {
		return nil, fmt.Errorf("too many limit value arguments - %s, can only have up to two, `soft[:hard]`", parts[1])
	}

	soft, err := strconv.ParseInt(limitVals[0], 10, 64)
	if err != nil {
		return nil, err
	}

	hard := soft // in case no hard was set
	if len(limitVals) == 2 {
		hard, err = strconv.ParseInt(limitVals[1], 10, 64)
	}
	if soft > hard {
		return nil, fmt.Errorf("ulimit soft limit must be less than or equal to hard limit: %d > %d", soft, hard)
	}

	return &Ulimit{Name: parts[0], Soft: soft, Hard: hard}, nil
}

func (u *Ulimit) GetRlimit() (*Rlimit, error) {
	t, exists := ulimitNameMapping[u.Name]
	if !exists {
		return nil, fmt.Errorf("invalid ulimit name %s", u.Name)
	}

	return &Rlimit{Type: t, Soft: uint64(u.Soft), Hard: uint64(u.Hard)}, nil
}

func (u *Ulimit) String() string {
	return fmt.Sprintf("%s=%d:%d", u.Name, u.Soft, u.Hard)
}
