package libcontainerd

import "github.com/docker/docker/pkg/locker"

type remote struct {
}

func (r *remote) Client(b Backend) (Client, error) {
	c := &client{
		clientCommon: clientCommon{
			backend:    b,
			containers: make(map[string]*container),
			locker:     locker.New(),
		},
	}
	return c, nil
}

// Cleanup is a no-op on Windows. It is here to implement the interface.
func (r *remote) Cleanup() {
}

func (r *remote) UpdateOptions(opts ...RemoteOption) error {
	return nil
}

// New creates a fresh instance of libcontainerd remote. On Windows,
// this is not used as there is no remote containerd process.
func New(_ string, _ ...RemoteOption) (Remote, error) {
	return &remote{}, nil
}

// WithLiveRestore is a noop on windows.
func WithLiveRestore(v bool) RemoteOption {
	return nil
}
