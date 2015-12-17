package meta

import (
	"errors"
	"fmt"
)

var (
	// ErrStoreOpen is returned when opening an already open store.
	ErrStoreOpen = newError("store already open")

	// ErrStoreClosed is returned when closing an already closed store.
	ErrStoreClosed = newError("raft store already closed")

	// ErrTooManyPeers is returned when more than 3 peers are used.
	ErrTooManyPeers = newError("too many peers; influxdb v0.9.0 is limited to 3 nodes in a cluster")
)

var (
	// ErrNodeExists is returned when creating an already existing node.
	ErrNodeExists = newError("node already exists")

	// ErrNodeNotFound is returned when mutating a node that doesn't exist.
	ErrNodeNotFound = newError("node not found")

	// ErrNodesRequired is returned when at least one node is required for an operation.
	// This occurs when creating a shard group.
	ErrNodesRequired = newError("at least one node required")

	// ErrNodeIDRequired is returned when using a zero node id.
	ErrNodeIDRequired = newError("node id must be greater than 0")

	// ErrNodeUnableToDropFinalNode is returned if the node being dropped is the last
	// node in the cluster
	ErrNodeUnableToDropFinalNode = newError("unable to drop the final node in a cluster")
)

var (
	// ErrDatabaseExists is returned when creating an already existing database.
	ErrDatabaseExists = newError("database already exists")

	// ErrDatabaseNameRequired is returned when creating a database without a name.
	ErrDatabaseNameRequired = newError("database name required")
)

var (
	// ErrRetentionPolicyExists is returned when creating an already existing policy.
	ErrRetentionPolicyExists = newError("retention policy already exists")

	// ErrRetentionPolicyDefault is returned when attempting a prohibited operation
	// on a default retention policy.
	ErrRetentionPolicyDefault = newError("retention policy is default")

	// ErrRetentionPolicyNameRequired is returned when creating a policy without a name.
	ErrRetentionPolicyNameRequired = newError("retention policy name required")

	// ErrRetentionPolicyNameExists is returned when renaming a policy to
	// the same name as another existing policy.
	ErrRetentionPolicyNameExists = newError("retention policy name already exists")

	// ErrRetentionPolicyDurationTooLow is returned when updating a retention
	// policy that has a duration lower than the allowed minimum.
	ErrRetentionPolicyDurationTooLow = newError(fmt.Sprintf("retention policy duration must be at least %s",
		MinRetentionPolicyDuration))

	// ErrReplicationFactorTooLow is returned when the replication factor is not in an
	// acceptable range.
	ErrReplicationFactorTooLow = newError("replication factor must be greater than 0")
)

var (
	// ErrShardGroupExists is returned when creating an already existing shard group.
	ErrShardGroupExists = newError("shard group already exists")

	// ErrShardGroupNotFound is returned when mutating a shard group that doesn't exist.
	ErrShardGroupNotFound = newError("shard group not found")

	// ErrShardNotReplicated is returned if the node requested to be dropped has
	// the last copy of a shard present and the force keyword was not used
	ErrShardNotReplicated = newError("shard not replicated")
)

var (
	// ErrContinuousQueryExists is returned when creating an already existing continuous query.
	ErrContinuousQueryExists = newError("continuous query already exists")

	// ErrContinuousQueryNotFound is returned when removing a continuous query that doesn't exist.
	ErrContinuousQueryNotFound = newError("continuous query not found")
)

var (
	// ErrSubscriptionExists is returned when creating an already existing subscription.
	ErrSubscriptionExists = newError("subscription already exists")

	// ErrSubscriptionNotFound is returned when removing a subscription that doesn't exist.
	ErrSubscriptionNotFound = newError("subscription not found")
)

var (
	// ErrUserExists is returned when creating an already existing user.
	ErrUserExists = newError("user already exists")

	// ErrUserNotFound is returned when mutating a user that doesn't exist.
	ErrUserNotFound = newError("user not found")

	// ErrUsernameRequired is returned when creating a user without a username.
	ErrUsernameRequired = newError("username required")
)

// errLookup stores a mapping of error strings to well defined error types.
var errLookup = make(map[string]error)

func newError(msg string) error {
	err := errors.New(msg)
	errLookup[err.Error()] = err
	return err
}

// lookupError returns a known error reference, if one exists.
// Otherwise returns err.
func lookupError(err error) error {
	if e, ok := errLookup[err.Error()]; ok {
		return e
	}
	return err
}
