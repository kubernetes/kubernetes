package linter

import (
	"fmt"
	"strconv"
	"strings"
)

type GoVersion struct {
	Major int
	Minor int
}

// GreaterOrEqual performs $v >= $other operation.
//
// In other words, it reports whether $v version constraint can use
// a feature from the $other Go version.
//
// As a special case, Major=0 covers all versions.
func (v GoVersion) GreaterOrEqual(other GoVersion) bool {
	if v.Major == 0 {
		return true
	}
	if v.Major == other.Major {
		return v.Minor >= other.Minor
	}
	return v.Major >= other.Major
}

func ParseGoVersion(version string) (GoVersion, error) {
	var result GoVersion
	version = strings.TrimPrefix(version, "go")
	if version == "" {
		return result, nil
	}
	parts := strings.Split(version, ".")
	if len(parts) != 2 {
		return result, fmt.Errorf("invalid Go version format: %s", version)
	}
	major, err := strconv.Atoi(parts[0])
	if err != nil {
		return result, fmt.Errorf("invalid major version part: %s: %w", parts[0], err)
	}
	minor, err := strconv.Atoi(parts[1])
	if err != nil {
		return result, fmt.Errorf("invalid minor version part: %s: %w", parts[1], err)
	}
	result.Major = major
	result.Minor = minor
	return result, nil
}
