package endpoints

import (
	"context"

	clientv3 "go.etcd.io/etcd/client/v3"
)

// Endpoint represents a single address the connection can be established with.
//
// Inspired by: https://pkg.go.dev/google.golang.org/grpc/resolver#Address.
// Please document etcd version since which version each field is supported.
type Endpoint struct {
	// Addr is the server address on which a connection will be established.
	// Since etcd 3.1
	Addr string

	// Metadata is the information associated with Addr, which may be used
	// to make load balancing decision.
	// Since etcd 3.1
	Metadata interface{}
}

type Operation uint8

const (
	// Add indicates an Endpoint is added.
	Add Operation = iota
	// Delete indicates an existing address is deleted.
	Delete
)

// Update describes a single edit action of an Endpoint.
type Update struct {
	// Op - action Add or Delete.
	Op       Operation
	Key      string
	Endpoint Endpoint
}

// WatchChannel is used to deliver notifications about endpoints updates.
type WatchChannel <-chan []*Update

// Key2EndpointMap maps etcd key into struct describing the endpoint.
type Key2EndpointMap map[string]Endpoint

// UpdateWithOpts describes endpoint update (add or delete) together
// with etcd options (e.g. to attach an endpoint to a lease).
type UpdateWithOpts struct {
	Update
	Opts []clientv3.OpOption
}

// NewAddUpdateOpts constructs UpdateWithOpts for endpoint registration.
func NewAddUpdateOpts(key string, endpoint Endpoint, opts ...clientv3.OpOption) *UpdateWithOpts {
	return &UpdateWithOpts{Update: Update{Op: Add, Key: key, Endpoint: endpoint}, Opts: opts}
}

// NewDeleteUpdateOpts constructs UpdateWithOpts for endpoint deletion.
func NewDeleteUpdateOpts(key string, opts ...clientv3.OpOption) *UpdateWithOpts {
	return &UpdateWithOpts{Update: Update{Op: Delete, Key: key}, Opts: opts}
}

// Manager can be used to add/remove & inspect endpoints stored in etcd for
// a particular target.
type Manager interface {
	// Update allows to atomically add/remove a few endpoints from etcd.
	Update(ctx context.Context, updates []*UpdateWithOpts) error

	// AddEndpoint registers a single endpoint in etcd.
	// For more advanced use-cases use the Update method.
	AddEndpoint(ctx context.Context, key string, endpoint Endpoint, opts ...clientv3.OpOption) error
	// DeleteEndpoint deletes a single endpoint stored in etcd.
	// For more advanced use-cases use the Update method.
	DeleteEndpoint(ctx context.Context, key string, opts ...clientv3.OpOption) error

	// List returns all the endpoints for the current target as a map.
	List(ctx context.Context) (Key2EndpointMap, error)
	// NewWatchChannel creates a channel that populates or endpoint updates.
	// Cancel the 'ctx' to close the watcher.
	NewWatchChannel(ctx context.Context) (WatchChannel, error)
}
