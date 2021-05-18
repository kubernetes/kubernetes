// +build linux

package fscommon

import (
	"errors"
	"fmt"
	"math"
	"strconv"
	"strings"
)

var (
	ErrNotValidFormat = errors.New("line is not a valid key value format")
)

// ParseUint converts a string to an uint64 integer.
// Negative values are returned at zero as, due to kernel bugs,
// some of the memory cgroup stats can be negative.
func ParseUint(s string, base, bitSize int) (uint64, error) {
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

// GetCgroupParamKeyValue parses a space-separated "name value" kind of cgroup
// parameter and returns its components. For example, "io_service_bytes 1234"
// will return as "io_service_bytes", 1234.
func GetCgroupParamKeyValue(t string) (string, uint64, error) {
	parts := strings.Fields(t)
	switch len(parts) {
	case 2:
		value, err := ParseUint(parts[1], 10, 64)
		if err != nil {
			return "", 0, fmt.Errorf("unable to convert to uint64: %v", err)
		}

		return parts[0], value, nil
	default:
		return "", 0, ErrNotValidFormat
	}
}

// GetCgroupParamUint reads a single uint64 value from the specified cgroup file.
// If the value read is "max", the math.MaxUint64 is returned.
func GetCgroupParamUint(path, file string) (uint64, error) {
	contents, err := GetCgroupParamString(path, file)
	if err != nil {
		return 0, err
	}
	contents = strings.TrimSpace(contents)
	if contents == "max" {
		return math.MaxUint64, nil
	}

	res, err := ParseUint(contents, 10, 64)
	if err != nil {
		return res, fmt.Errorf("unable to parse file %q", path+"/"+file)
	}
	return res, nil
}

// GetCgroupParamInt reads a single int64 value from specified cgroup file.
// If the value read is "max", the math.MaxInt64 is returned.
func GetCgroupParamInt(path, file string) (int64, error) {
	contents, err := ReadFile(path, file)
	if err != nil {
		return 0, err
	}
	contents = strings.TrimSpace(contents)
	if contents == "max" {
		return math.MaxInt64, nil
	}

	res, err := strconv.ParseInt(contents, 10, 64)
	if err != nil {
		return res, fmt.Errorf("unable to parse %q as a int from Cgroup file %q", contents, path+"/"+file)
	}
	return res, nil
}

// GetCgroupParamString reads a string from the specified cgroup file.
func GetCgroupParamString(path, file string) (string, error) {
	contents, err := ReadFile(path, file)
	if err != nil {
		return "", err
	}

	return strings.TrimSpace(contents), nil
}
