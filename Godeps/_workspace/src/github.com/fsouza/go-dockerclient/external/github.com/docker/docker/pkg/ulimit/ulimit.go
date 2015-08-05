// Package ulimit provides structure and helper function to parse and represent
// resource limits (Rlimit and Ulimit, its human friendly version).
package ulimit

import (
	"fmt"
	"strconv"
	"strings"
)

// Ulimit is a human friendly version of Rlimit.
type Ulimit struct {
	Name string
	Hard int64
	Soft int64
}

// Rlimit specifies the resource limits, such as max open files.
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
	rlimitAs         = 9
	rlimitCore       = 4
	rlimitCPU        = 0
	rlimitData       = 2
	rlimitFsize      = 1
	rlimitLocks      = 10
	rlimitMemlock    = 8
	rlimitMsgqueue   = 12
	rlimitNice       = 13
	rlimitNofile     = 7
	rlimitNproc      = 6
	rlimitRss        = 5
	rlimitRtprio     = 14
	rlimitRttime     = 15
	rlimitSigpending = 11
	rlimitStack      = 3
)

var ulimitNameMapping = map[string]int{
	//"as":         rlimitAs, // Disabled since this doesn't seem usable with the way Docker inits a container.
	"core":       rlimitCore,
	"cpu":        rlimitCPU,
	"data":       rlimitData,
	"fsize":      rlimitFsize,
	"locks":      rlimitLocks,
	"memlock":    rlimitMemlock,
	"msgqueue":   rlimitMsgqueue,
	"nice":       rlimitNice,
	"nofile":     rlimitNofile,
	"nproc":      rlimitNproc,
	"rss":        rlimitRss,
	"rtprio":     rlimitRtprio,
	"rttime":     rlimitRttime,
	"sigpending": rlimitSigpending,
	"stack":      rlimitStack,
}

// Parse parses and returns a Ulimit from the specified string.
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

// GetRlimit returns the RLimit corresponding to Ulimit.
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
