package fscommon

import (
	"errors"
	"fmt"
	"math"
	"path"
	"strconv"
	"strings"

	"github.com/opencontainers/runc/libcontainer/cgroups"
)

var (
	// Deprecated: use cgroups.OpenFile instead.
	OpenFile = cgroups.OpenFile
	// Deprecated: use cgroups.ReadFile instead.
	ReadFile = cgroups.ReadFile
	// Deprecated: use cgroups.WriteFile instead.
	WriteFile = cgroups.WriteFile
)

// ParseError records a parse error details, including the file path.
type ParseError struct {
	Path string
	File string
	Err  error
}

func (e *ParseError) Error() string {
	return "unable to parse " + path.Join(e.Path, e.File) + ": " + e.Err.Error()
}

func (e *ParseError) Unwrap() error { return e.Err }

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
		} else if errors.Is(intErr, strconv.ErrRange) && intValue < 0 {
			return 0, nil
		}

		return value, err
	}

	return value, nil
}

// ParseKeyValue parses a space-separated "name value" kind of cgroup
// parameter and returns its key as a string, and its value as uint64
// (ParseUint is used to convert the value). For example,
// "io_service_bytes 1234" will be returned as "io_service_bytes", 1234.
func ParseKeyValue(t string) (string, uint64, error) {
	parts := strings.SplitN(t, " ", 3)
	if len(parts) != 2 {
		return "", 0, fmt.Errorf("line %q is not in key value format", t)
	}

	value, err := ParseUint(parts[1], 10, 64)
	if err != nil {
		return "", 0, err
	}

	return parts[0], value, nil
}

// GetValueByKey reads a key-value pairs from the specified cgroup file,
// and returns a value of the specified key. ParseUint is used for value
// conversion.
func GetValueByKey(path, file, key string) (uint64, error) {
	content, err := cgroups.ReadFile(path, file)
	if err != nil {
		return 0, err
	}

	lines := strings.Split(content, "\n")
	for _, line := range lines {
		arr := strings.Split(line, " ")
		if len(arr) == 2 && arr[0] == key {
			val, err := ParseUint(arr[1], 10, 64)
			if err != nil {
				err = &ParseError{Path: path, File: file, Err: err}
			}
			return val, err
		}
	}

	return 0, nil
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
		return res, &ParseError{Path: path, File: file, Err: err}
	}
	return res, nil
}

// GetCgroupParamInt reads a single int64 value from specified cgroup file.
// If the value read is "max", the math.MaxInt64 is returned.
func GetCgroupParamInt(path, file string) (int64, error) {
	contents, err := cgroups.ReadFile(path, file)
	if err != nil {
		return 0, err
	}
	contents = strings.TrimSpace(contents)
	if contents == "max" {
		return math.MaxInt64, nil
	}

	res, err := strconv.ParseInt(contents, 10, 64)
	if err != nil {
		return res, &ParseError{Path: path, File: file, Err: err}
	}
	return res, nil
}

// GetCgroupParamString reads a string from the specified cgroup file.
func GetCgroupParamString(path, file string) (string, error) {
	contents, err := cgroups.ReadFile(path, file)
	if err != nil {
		return "", err
	}

	return strings.TrimSpace(contents), nil
}
