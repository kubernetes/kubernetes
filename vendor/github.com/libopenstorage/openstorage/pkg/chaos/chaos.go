package chaos

import (
	"errors"
	"fmt"
	"math/rand"
	"sync/atomic"
	"time"
)

// When defines when a chaos point is triggered
type When int

const (
	// Once chaos at the first call to ChaosNow.
	Once When = 1 << iota
	// Random chaos at a random number of calls to ChaosNow.
	Random
)

// Action to take when a chaos point is triggered.
type Action int

const (
	// Crash at a call to Now().
	Crash Action = 1 << iota
	// Error out at calll to Now();
	Error
)

// ID identifies a chaos point.
type ID uint32

// Chaos represents an instance of a chaos point
type Chaos struct {
	ID      ID
	Pkg     string
	Fn      string
	Desc    string
	Enabled bool
	What    Action
	When    When
	Count   int
}

var (
	activated bool
	chaos     map[ID]*Chaos
	count     int32
	r         *rand.Rand
	// ErrNoEnt is generated on an unknown chaos ID.
	ErrNoEnt = errors.New("ID does not exist")
	// ErrChaos is generated when Action is set to Error
	ErrChaos = errors.New("Chaos generated error")
)

// Activate activates chaos points in the system.
func Activate(activate bool) {
	activated = activate
}

// Add new chaos point. The ID returned can be used to perform operations on this Chaos Point.
func Add(pkg string, fn string, desc string) ID {
	id := ID(atomic.AddInt32(&count, 1))
	chaos[id] = &Chaos{ID: id, Pkg: pkg, Fn: fn, Desc: desc, Enabled: false}
	return id
}

// Enumerate all chaos points in the system for specified package.
// If the pkg is "" enumerate all chaos points.
func Enumerate(pkg string) []Chaos {

	ko := make([]Chaos, 0, 10)
	for _, v := range chaos {
		if pkg == "" || pkg == v.Pkg {
			ko = append(ko, *v)
		}
	}
	return ko
}

// Enable chaos point identified by ID.
func Enable(id ID, when When, what Action) error {
	if v, ok := chaos[id]; !ok {
		return ErrNoEnt
	} else {
		v.Enabled = true
		v.When = when
		v.What = what
	}
	return nil
}

// Disable chaos point identified by ID.
func Disable(id ID) error {
	if v, ok := chaos[id]; !ok {
		return ErrNoEnt
	} else {
		v.Enabled = false
	}
	return nil
}

// Now will trigger chaos point if it is enabled.
func Now(id ID) error {
	if !activated {
		return nil
	}
	if v, ok := chaos[id]; ok && v.Enabled {
		v.Count++
		if v.When == Once ||
			(v.Count%(r.Int()%10)) == 0 {
			if v.What == Error {
				return ErrChaos
			}
		} else {
			panic(fmt.Sprintf("Chaos triggered panic"))
		}
	}
	return nil
}

func init() {
	r = rand.New(rand.NewSource(time.Now().UnixNano()))
	activated = false
	chaos = make(map[ID]*Chaos)
	count = -1
}
