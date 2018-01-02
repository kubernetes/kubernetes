package docker

import (
	"sync"

	"github.com/containerd/containerd/content"
	"github.com/containerd/containerd/errdefs"
	"github.com/pkg/errors"
)

// Status of a content operation
type Status struct {
	content.Status

	// UploadUUID is used by the Docker registry to reference blob uploads
	UploadUUID string
}

// StatusTracker to track status of operations
type StatusTracker interface {
	GetStatus(string) (Status, error)
	SetStatus(string, Status)
}

type memoryStatusTracker struct {
	statuses map[string]Status
	m        sync.Mutex
}

// NewInMemoryTracker returns a StatusTracker that tracks content status in-memory
func NewInMemoryTracker() StatusTracker {
	return &memoryStatusTracker{
		statuses: map[string]Status{},
	}
}

func (t *memoryStatusTracker) GetStatus(ref string) (Status, error) {
	t.m.Lock()
	defer t.m.Unlock()
	status, ok := t.statuses[ref]
	if !ok {
		return Status{}, errors.Wrapf(errdefs.ErrNotFound, "status for ref %v", ref)
	}
	return status, nil
}

func (t *memoryStatusTracker) SetStatus(ref string, status Status) {
	t.m.Lock()
	t.statuses[ref] = status
	t.m.Unlock()
}
