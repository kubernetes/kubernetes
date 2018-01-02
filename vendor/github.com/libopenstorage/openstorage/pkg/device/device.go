package device

import (
	"errors"
	"fmt"
	"strings"
	"sync"
)

// Ops defines the interface to keep track of attached devices.
type Ops interface {
	// String representation of the mount table
	String() string
	// Assigns new path
	Assign() (string, error)
	// Releases path to available devices
	Release(device string) error
}

var (
	// ErrEnospc is returned if no devices can be allocated.
	ErrEnospc = errors.New("No free device IDs")
	// ErrEinval is returned if a device string is invalid
	ErrEinval = errors.New("Invalid device")
)

// SingleLetter defines a new device letter
type SingleLetter struct {
	sync.Mutex
	devices   string
	devPrefix string
}

// String is a description of this device.
func (s *SingleLetter) String() string {
	return "SingleLetter"
}

// NewSingleLetter instance of Matrix
func NewSingleLetter(devPrefix, devices string) (*SingleLetter, error) {
	s := &SingleLetter{
		devPrefix: devPrefix,
		devices:   devices,
	}
	return s, nil
}

// Assign new device letter
func (s *SingleLetter) Assign() (string, error) {
	s.Lock()
	defer s.Unlock()
	if len(s.devices) == 0 {
		return "", fmt.Errorf("No free device IDs")
	}
	device := s.devPrefix + s.devices[:1]
	s.devices = s.devices[1:]
	return device, nil
}

// Release device letter to devices pool.
func (s *SingleLetter) Release(dev string) error {
	s.Lock()
	defer s.Unlock()
	if !strings.HasPrefix(dev, s.devPrefix) {
		return ErrEinval
	}
	dev = dev[len(s.devPrefix):]
	s.devices += dev
	return nil
}
