// +build linux

package fs

import (
	"errors"
	"fmt"
	"io/ioutil"
	"path/filepath"
	"strconv"
	"strings"
)

var (
	ErrNotSupportStat = errors.New("stats are not supported for subsystem")
	ErrNotValidFormat = errors.New("line is not a valid key value format")
)

// Saturates negative values at zero and returns a uint64.
// Due to kernel bugs, some of the memory cgroup stats can be negative.
func parseUint(s string, base, bitSize int) (uint64, error) {
	value, err := strconv.ParseUint(s, base, bitSize)
	if err != nil {
		intValue, intErr := strconv.ParseInt(s, base, bitSize)
		// 1. Handle negative values greater than MinInt64 (and)
		// 2. Handle negative values lesser than MinInt64
		if intErr == nil && intValue < 0 {
			return 0, nil
		} else if intErr != nil && intErr.(*strconv.NumError).Err == strconv.ErrRange && intValue < 0 {
			return 0, nil
		}

		return value, err
	}

	return value, nil
}

// Parses a cgroup param and returns as name, value
//  i.e. "io_service_bytes 1234" will return as io_service_bytes, 1234
func getCgroupParamKeyValue(t string) (string, uint64, error) {
	parts := strings.Fields(t)
	switch len(parts) {
	case 2:
		value, err := parseUint(parts[1], 10, 64)
		if err != nil {
			return "", 0, fmt.Errorf("Unable to convert param value (%q) to uint64: %v", parts[1], err)
		}

		return parts[0], value, nil
	default:
		return "", 0, ErrNotValidFormat
	}
}

// Gets a single uint64 value from the specified cgroup file.
func getCgroupParamUint(cgroupPath, cgroupFile string) (uint64, error) {
	contents, err := ioutil.ReadFile(filepath.Join(cgroupPath, cgroupFile))
	if err != nil {
		return 0, err
	}

	return parseUint(strings.TrimSpace(string(contents)), 10, 64)
}

// Gets a string value from the specified cgroup file
func getCgroupParamString(cgroupPath, cgroupFile string) (string, error) {
	contents, err := ioutil.ReadFile(filepath.Join(cgroupPath, cgroupFile))
	if err != nil {
		return "", err
	}

	return strings.TrimSpace(string(contents)), nil
}
