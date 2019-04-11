// Package parsers provides helper functions to parse and validate different type
// of string. It can be hosts, unix addresses, tcp addresses, filters, kernel
// operating system versions.
package parsers // import "github.com/docker/docker/pkg/parsers"

import (
	"fmt"
	"strconv"
	"strings"
)

// ParseKeyValueOpt parses and validates the specified string as a key/value pair (key=value)
func ParseKeyValueOpt(opt string) (string, string, error) {
	parts := strings.SplitN(opt, "=", 2)
	if len(parts) != 2 {
		return "", "", fmt.Errorf("Unable to parse key/value option: %s", opt)
	}
	return strings.TrimSpace(parts[0]), strings.TrimSpace(parts[1]), nil
}

// ParseUintListMaximum parses and validates the specified string as the value
// found in some cgroup file (e.g. `cpuset.cpus`, `cpuset.mems`), which could be
// one of the formats below. Note that duplicates are actually allowed in the
// input string. It returns a `map[int]bool` with available elements from `val`
// set to `true`. Values larger than `maximum` cause an error if max is non zero,
// in order to stop the map becoming excessively large.
// Supported formats:
//     7
//     1-6
//     0,3-4,7,8-10
//     0-0,0,1-7
//     03,1-3      <- this is gonna get parsed as [1,2,3]
//     3,2,1
//     0-2,3,1
func ParseUintListMaximum(val string, maximum int) (map[int]bool, error) {
	return parseUintList(val, maximum)
}

// ParseUintList parses and validates the specified string as the value
// found in some cgroup file (e.g. `cpuset.cpus`, `cpuset.mems`), which could be
// one of the formats below. Note that duplicates are actually allowed in the
// input string. It returns a `map[int]bool` with available elements from `val`
// set to `true`.
// Supported formats:
//     7
//     1-6
//     0,3-4,7,8-10
//     0-0,0,1-7
//     03,1-3      <- this is gonna get parsed as [1,2,3]
//     3,2,1
//     0-2,3,1
func ParseUintList(val string) (map[int]bool, error) {
	return parseUintList(val, 0)
}

func parseUintList(val string, maximum int) (map[int]bool, error) {
	if val == "" {
		return map[int]bool{}, nil
	}

	availableInts := make(map[int]bool)
	split := strings.Split(val, ",")
	errInvalidFormat := fmt.Errorf("invalid format: %s", val)

	for _, r := range split {
		if !strings.Contains(r, "-") {
			v, err := strconv.Atoi(r)
			if err != nil {
				return nil, errInvalidFormat
			}
			if maximum != 0 && v > maximum {
				return nil, fmt.Errorf("value of out range, maximum is %d", maximum)
			}
			availableInts[v] = true
		} else {
			split := strings.SplitN(r, "-", 2)
			min, err := strconv.Atoi(split[0])
			if err != nil {
				return nil, errInvalidFormat
			}
			max, err := strconv.Atoi(split[1])
			if err != nil {
				return nil, errInvalidFormat
			}
			if max < min {
				return nil, errInvalidFormat
			}
			if maximum != 0 && max > maximum {
				return nil, fmt.Errorf("value of out range, maximum is %d", maximum)
			}
			for i := min; i <= max; i++ {
				availableInts[i] = true
			}
		}
	}
	return availableInts, nil
}
