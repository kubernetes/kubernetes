package internal

// Operation describes action performed on endpoint (addition vs deletion).
// Must stay JSON-format compatible with:
// https://pkg.go.dev/google.golang.org/grpc@v1.29.1/naming#Operation
type Operation uint8

const (
	// Add indicates a new address is added.
	Add Operation = iota
	// Delete indicates an existing address is deleted.
	Delete
)

// Update defines a persistent (JSON marshalled) format representing
// endpoint within the etcd storage.
//
// As the format can be persisted by one version of etcd client library and
// read by other the format must be kept backward compatible and
// in particular must be superset of the grpc(<=1.29.1) naming.Update structure:
// https://pkg.go.dev/google.golang.org/grpc@v1.29.1/naming#Update
//
// Please document since which version of etcd-client given property is supported.
// Please keep the naming consistent with e.g. https://pkg.go.dev/google.golang.org/grpc/resolver#Address.
//
// Notice that it is not valid having both empty string Addr and nil Metadata in an Update.
type Update struct {
	// Op indicates the operation of the update.
	// Since etcd 3.1.
	Op Operation
	// Addr is the updated address. It is empty string if there is no address update.
	// Since etcd 3.1.
	Addr string
	// Metadata is the updated metadata. It is nil if there is no metadata update.
	// Metadata is not required for a custom naming implementation.
	// Since etcd 3.1.
	Metadata interface{}
}
