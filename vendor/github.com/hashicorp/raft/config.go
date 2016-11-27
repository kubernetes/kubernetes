package raft

import (
	"fmt"
	"io"
	"log"
	"time"
)

// Config provides any necessary configuration to
// the Raft server
type Config struct {
	// HeartbeatTimeout specifies the time in follower state without
	// a leader before we attempt an election.
	HeartbeatTimeout time.Duration

	// ElectionTimeout specifies the time in candidate state without
	// a leader before we attempt an election.
	ElectionTimeout time.Duration

	// CommitTimeout controls the time without an Apply() operation
	// before we heartbeat to ensure a timely commit. Due to random
	// staggering, may be delayed as much as 2x this value.
	CommitTimeout time.Duration

	// MaxAppendEntries controls the maximum number of append entries
	// to send at once. We want to strike a balance between efficiency
	// and avoiding waste if the follower is going to reject because of
	// an inconsistent log.
	MaxAppendEntries int

	// If we are a member of a cluster, and RemovePeer is invoked for the
	// local node, then we forget all peers and transition into the follower state.
	// If ShutdownOnRemove is is set, we additional shutdown Raft. Otherwise,
	// we can become a leader of a cluster containing only this node.
	ShutdownOnRemove bool

	// DisableBootstrapAfterElect is used to turn off EnableSingleNode
	// after the node is elected. This is used to prevent self-election
	// if the node is removed from the Raft cluster via RemovePeer. Setting
	// it to false will keep the bootstrap mode, allowing the node to self-elect
	// and potentially bootstrap a separate cluster.
	DisableBootstrapAfterElect bool

	// TrailingLogs controls how many logs we leave after a snapshot. This is
	// used so that we can quickly replay logs on a follower instead of being
	// forced to send an entire snapshot.
	TrailingLogs uint64

	// SnapshotInterval controls how often we check if we should perform a snapshot.
	// We randomly stagger between this value and 2x this value to avoid the entire
	// cluster from performing a snapshot at once.
	SnapshotInterval time.Duration

	// SnapshotThreshold controls how many outstanding logs there must be before
	// we perform a snapshot. This is to prevent excessive snapshots when we can
	// just replay a small set of logs.
	SnapshotThreshold uint64

	// EnableSingleNode allows for a single node mode of operation. This
	// is false by default, which prevents a lone node from electing itself.
	// leader.
	EnableSingleNode bool

	// LeaderLeaseTimeout is used to control how long the "lease" lasts
	// for being the leader without being able to contact a quorum
	// of nodes. If we reach this interval without contact, we will
	// step down as leader.
	LeaderLeaseTimeout time.Duration

	// StartAsLeader forces Raft to start in the leader state. This should
	// never be used except for testing purposes, as it can cause a split-brain.
	StartAsLeader bool

	// NotifyCh is used to provide a channel that will be notified of leadership
	// changes. Raft will block writing to this channel, so it should either be
	// buffered or aggressively consumed.
	NotifyCh chan<- bool

	// LogOutput is used as a sink for logs, unless Logger is specified.
	// Defaults to os.Stderr.
	LogOutput io.Writer

	// Logger is a user-provided logger. If nil, a logger writing to LogOutput
	// is used.
	Logger *log.Logger
}

// DefaultConfig returns a Config with usable defaults.
func DefaultConfig() *Config {
	return &Config{
		HeartbeatTimeout:           1000 * time.Millisecond,
		ElectionTimeout:            1000 * time.Millisecond,
		CommitTimeout:              50 * time.Millisecond,
		MaxAppendEntries:           64,
		ShutdownOnRemove:           true,
		DisableBootstrapAfterElect: true,
		TrailingLogs:               10240,
		SnapshotInterval:           120 * time.Second,
		SnapshotThreshold:          8192,
		EnableSingleNode:           false,
		LeaderLeaseTimeout:         500 * time.Millisecond,
	}
}

// ValidateConfig is used to validate a sane configuration
func ValidateConfig(config *Config) error {
	if config.HeartbeatTimeout < 5*time.Millisecond {
		return fmt.Errorf("Heartbeat timeout is too low")
	}
	if config.ElectionTimeout < 5*time.Millisecond {
		return fmt.Errorf("Election timeout is too low")
	}
	if config.CommitTimeout < time.Millisecond {
		return fmt.Errorf("Commit timeout is too low")
	}
	if config.MaxAppendEntries <= 0 {
		return fmt.Errorf("MaxAppendEntries must be positive")
	}
	if config.MaxAppendEntries > 1024 {
		return fmt.Errorf("MaxAppendEntries is too large")
	}
	if config.SnapshotInterval < 5*time.Millisecond {
		return fmt.Errorf("Snapshot interval is too low")
	}
	if config.LeaderLeaseTimeout < 5*time.Millisecond {
		return fmt.Errorf("Leader lease timeout is too low")
	}
	if config.LeaderLeaseTimeout > config.HeartbeatTimeout {
		return fmt.Errorf("Leader lease timeout cannot be larger than heartbeat timeout")
	}
	if config.ElectionTimeout < config.HeartbeatTimeout {
		return fmt.Errorf("Election timeout must be equal or greater than Heartbeat Timeout")
	}
	return nil
}
