package models

import (
	"errors"
	"strings"
)

// ConsistencyLevel represent a required replication criteria before a write can
// be returned as successful
type ConsistencyLevel int

const (
	// ConsistencyLevelAny allows for hinted hand off, potentially no write happened yet
	ConsistencyLevelAny ConsistencyLevel = iota

	// ConsistencyLevelOne requires at least one data node acknowledged a write
	ConsistencyLevelOne

	// ConsistencyLevelQuorum requires a quorum of data nodes to acknowledge a write
	ConsistencyLevelQuorum

	// ConsistencyLevelAll requires all data nodes to acknowledge a write
	ConsistencyLevelAll
)

var (
	// ErrInvalidConsistencyLevel is returned when parsing the string version
	// of a consistency level.
	ErrInvalidConsistencyLevel = errors.New("invalid consistency level")
)

// ParseConsistencyLevel converts a consistency level string to the corresponding ConsistencyLevel const
func ParseConsistencyLevel(level string) (ConsistencyLevel, error) {
	switch strings.ToLower(level) {
	case "any":
		return ConsistencyLevelAny, nil
	case "one":
		return ConsistencyLevelOne, nil
	case "quorum":
		return ConsistencyLevelQuorum, nil
	case "all":
		return ConsistencyLevelAll, nil
	default:
		return 0, ErrInvalidConsistencyLevel
	}
}
