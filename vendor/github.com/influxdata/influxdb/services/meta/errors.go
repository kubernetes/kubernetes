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
)

var (
	// ErrDatabaseExists is returned when creating an already existing database.
	ErrDatabaseExists = errors.New("database already exists")

	// ErrDatabaseNotExists is returned when operating on a not existing database.
	ErrDatabaseNotExists = errors.New("database does not exist")

	// ErrDatabaseNameRequired is returned when creating a database without a name.
	ErrDatabaseNameRequired = errors.New("database name required")
)

var (
	// ErrRetentionPolicyExists is returned when creating an already existing policy.
	ErrRetentionPolicyExists = errors.New("retention policy already exists")

	// ErrRetentionPolicyNotFound is returned when an expected policy wasn't found.
	ErrRetentionPolicyNotFound = errors.New("retention policy not found")

	// ErrRetentionPolicyDefault is returned when attempting a prohibited operation
	// on a default retention policy.
	ErrRetentionPolicyDefault = errors.New("retention policy is default")

	// ErrRetentionPolicyRequired is returned when a retention policy is required
	// by an operation, but a nil policy was passed.
	ErrRetentionPolicyRequired = errors.New("retention policy required")

	// ErrRetentionPolicyNameRequired is returned when creating a policy without a name.
	ErrRetentionPolicyNameRequired = errors.New("retention policy name required")

	// ErrRetentionPolicyNameExists is returned when renaming a policy to
	// the same name as another existing policy.
	ErrRetentionPolicyNameExists = errors.New("retention policy name already exists")

	// ErrRetentionPolicyDurationTooLow is returned when updating a retention
	// policy that has a duration lower than the allowed minimum.
	ErrRetentionPolicyDurationTooLow = errors.New(fmt.Sprintf("retention policy duration must be at least %s",
		MinRetentionPolicyDuration))

	// ErrRetentionPolicyConflict is returned when creating a retention policy conflicts
	// with an existing policy.
	ErrRetentionPolicyConflict = errors.New("retention policy conflicts with an existing policy")

	// ErrIncompatibleDurations is returned when creating or updating a
	// retention policy that has a duration lower than the current shard
	// duration.
	ErrIncompatibleDurations = errors.New("retention policy duration must be greater than the shard duration")

	// ErrReplicationFactorTooLow is returned when the replication factor is not in an
	// acceptable range.
	ErrReplicationFactorTooLow = errors.New("replication factor must be greater than 0")
)

var (
	// ErrShardGroupExists is returned when creating an already existing shard group.
	ErrShardGroupExists = errors.New("shard group already exists")

	// ErrShardGroupNotFound is returned when mutating a shard group that doesn't exist.
	ErrShardGroupNotFound = errors.New("shard group not found")

	// ErrShardNotReplicated is returned if the node requested to be dropped has
	// the last copy of a shard present and the force keyword was not used
	ErrShardNotReplicated = errors.New("shard not replicated")
)

var (
	// ErrContinuousQueryExists is returned when creating an already existing continuous query.
	ErrContinuousQueryExists = errors.New("continuous query already exists")

	// ErrContinuousQueryNotFound is returned when removing a continuous query that doesn't exist.
	ErrContinuousQueryNotFound = errors.New("continuous query not found")
)

var (
	// ErrSubscriptionExists is returned when creating an already existing subscription.
	ErrSubscriptionExists = errors.New("subscription already exists")

	// ErrSubscriptionNotFound is returned when removing a subscription that doesn't exist.
	ErrSubscriptionNotFound = errors.New("subscription not found")
)

var (
	// ErrUserExists is returned when creating an already existing user.
	ErrUserExists = errors.New("user already exists")

	// ErrUserNotFound is returned when mutating a user that doesn't exist.
	ErrUserNotFound = errors.New("user not found")

	// ErrUsernameRequired is returned when creating a user without a username.
	ErrUsernameRequired = errors.New("username required")

	// ErrAuthenticate is returned when authentication fails.
	ErrAuthenticate = errors.New("authentication failed")
)
