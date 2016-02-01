package meta

import (
	"errors"
	"fmt"
)

var (
	// ErrStoreOpen is returned when opening an already open store.
	ErrStoreOpen = errors.New("store already open")

	// ErrStoreClosed is returned when closing an already closed store.
	ErrStoreClosed = errors.New("raft store already closed")

	// ErrTooManyPeers is returned when more than 3 peers are used.
	ErrTooManyPeers = errors.New("too many peers; influxdb v0.9.0 is limited to 3 nodes in a cluster")
)

var (
	// ErrNodeExists is returned when creating an already existing node.
	ErrNodeExists = errors.New("node already exists")

	// ErrNodeNotFound is returned when mutating a node that doesn't exist.
	ErrNodeNotFound = errors.New("node not found")

	// ErrNodesRequired is returned when at least one node is required for an operation.
	// This occurs when creating a shard group.
	ErrNodesRequired = errors.New("at least one node required")
)

var (
	// ErrDatabaseExists is returned when creating an already existing database.
	ErrDatabaseExists = errors.New("database already exists")

	// ErrDatabaseNotFound is returned when mutating a database that doesn't exist.
	ErrDatabaseNotFound = errors.New("database not found")

	// ErrDatabaseNameRequired is returned when creating a database without a name.
	ErrDatabaseNameRequired = errors.New("database name required")
)

var (
	// ErrRetentionPolicyExists is returned when creating an already existing policy.
	ErrRetentionPolicyExists = errors.New("retention policy already exists")

	// ErrRetentionPolicyNotFound is returned when mutating a policy that doesn't exist.
	ErrRetentionPolicyNotFound = errors.New("retention policy not found")

	// ErrRetentionPolicyNameRequired is returned when creating a policy without a name.
	ErrRetentionPolicyNameRequired = errors.New("retention policy name required")

	// ErrRetentionPolicyNameExists is returned when renaming a policy to
	// the same name as another existing policy.
	ErrRetentionPolicyNameExists = errors.New("retention policy name already exists")

	// ErrRetentionPolicyDurationTooLow is returned when updating a retention
	// policy that has a duration lower than the allowed minimum.
	ErrRetentionPolicyDurationTooLow = errors.New(fmt.Sprintf("retention policy duration must be at least %s",
		RetentionPolicyMinDuration))

	// ErrReplicationFactorMismatch is returned when the replication factor
	// does not match the number of nodes in the cluster. This is a temporary
	// restriction until v0.9.1 is released.
	ErrReplicationFactorMismatch = errors.New("replication factor must match cluster size; this limitation will be lifted in v0.9.1")
)

var (
	// ErrShardGroupExists is returned when creating an already existing shard group.
	ErrShardGroupExists = errors.New("shard group already exists")

	// ErrShardGroupNotFound is returned when mutating a shard group that doesn't exist.
	ErrShardGroupNotFound = errors.New("shard group not found")
)

var (
	// ErrContinuousQueryExists is returned when creating an already existing continuous query.
	ErrContinuousQueryExists = errors.New("continuous query already exists")

	// ErrContinuousQueryNotFound is returned when removing a continuous query that doesn't exist.
	ErrContinuousQueryNotFound = errors.New("continuous query not found")
)

var (
	// ErrUserExists is returned when creating an already existing user.
	ErrUserExists = errors.New("user already exists")

	// ErrUserNotFound is returned when mutating a user that doesn't exist.
	ErrUserNotFound = errors.New("user not found")

	// ErrUsernameRequired is returned when creating a user without a username.
	ErrUsernameRequired = errors.New("username required")
)

var errs = [...]error{
	ErrStoreOpen, ErrStoreClosed,
	ErrNodeExists, ErrNodeNotFound,
	ErrDatabaseExists, ErrDatabaseNotFound, ErrDatabaseNameRequired,
}

// errLookup stores a mapping of error strings to well defined error types.
var errLookup = make(map[string]error)

func init() {
	for _, err := range errs {
		errLookup[err.Error()] = err
	}
}

// lookupError returns a known error reference, if one exists.
// Otherwise returns err.
func lookupError(err error) error {
	if e, ok := errLookup[err.Error()]; ok {
		return e
	}
	return err
}
