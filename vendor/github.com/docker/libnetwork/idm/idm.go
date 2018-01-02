// Package idm manages reservation/release of numerical ids from a configured set of contiguous ids
package idm

import (
	"errors"
	"fmt"

	"github.com/docker/libnetwork/bitseq"
	"github.com/docker/libnetwork/datastore"
)

// Idm manages the reservation/release of numerical ids from a contiguous set
type Idm struct {
	start  uint64
	end    uint64
	handle *bitseq.Handle
}

// New returns an instance of id manager for a [start,end] set of numerical ids
func New(ds datastore.DataStore, id string, start, end uint64) (*Idm, error) {
	if id == "" {
		return nil, errors.New("Invalid id")
	}
	if end <= start {
		return nil, fmt.Errorf("Invalid set range: [%d, %d]", start, end)
	}

	h, err := bitseq.NewHandle("idm", ds, id, 1+end-start)
	if err != nil {
		return nil, fmt.Errorf("failed to initialize bit sequence handler: %s", err.Error())
	}

	return &Idm{start: start, end: end, handle: h}, nil
}

// GetID returns the first available id in the set
func (i *Idm) GetID() (uint64, error) {
	if i.handle == nil {
		return 0, errors.New("ID set is not initialized")
	}
	ordinal, err := i.handle.SetAny()
	return i.start + ordinal, err
}

// GetSpecificID tries to reserve the specified id
func (i *Idm) GetSpecificID(id uint64) error {
	if i.handle == nil {
		return errors.New("ID set is not initialized")
	}

	if id < i.start || id > i.end {
		return errors.New("Requested id does not belong to the set")
	}

	return i.handle.Set(id - i.start)
}

// GetIDInRange returns the first available id in the set within a [start,end] range
func (i *Idm) GetIDInRange(start, end uint64) (uint64, error) {
	if i.handle == nil {
		return 0, errors.New("ID set is not initialized")
	}

	if start < i.start || end > i.end {
		return 0, errors.New("Requested range does not belong to the set")
	}

	ordinal, err := i.handle.SetAnyInRange(start-i.start, end-i.start)

	return i.start + ordinal, err
}

// Release releases the specified id
func (i *Idm) Release(id uint64) {
	i.handle.Unset(id - i.start)
}
