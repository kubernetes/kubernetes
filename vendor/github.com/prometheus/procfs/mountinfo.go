// Copyright 2019 The Prometheus Authors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package procfs

import (
	"bufio"
	"fmt"
	"io"
	"os"
	"strconv"
	"strings"
)

var validOptionalFields = map[string]bool{
	"shared":         true,
	"master":         true,
	"propagate_from": true,
	"unbindable":     true,
}

// A MountInfo is a type that describes the details, options
// for each mount, parsed from /proc/self/mountinfo.
// The fields described in each entry of /proc/self/mountinfo
// is described in the following man page.
// http://man7.org/linux/man-pages/man5/proc.5.html
type MountInfo struct {
	// Unique Id for the mount
	MountId int
	// The Id of the parent mount
	ParentId int
	// The value of `st_dev` for the files on this FS
	MajorMinorVer string
	// The pathname of the directory in the FS that forms
	// the root for this mount
	Root string
	// The pathname of the mount point relative to the root
	MountPoint string
	// Mount options
	Options map[string]string
	// Zero or more optional fields
	OptionalFields map[string]string
	// The Filesystem type
	FSType string
	// FS specific information or "none"
	Source string
	// Superblock options
	SuperOptions map[string]string
}

// Returns part of the mountinfo line, if it exists, else an empty string.
func getStringSliceElement(parts []string, idx int, defaultValue string) string {
	if idx >= len(parts) {
		return defaultValue
	}
	return parts[idx]
}

// Reads each line of the mountinfo file, and returns a list of formatted MountInfo structs.
func parseMountInfo(r io.Reader) ([]*MountInfo, error) {
	mounts := []*MountInfo{}
	scanner := bufio.NewScanner(r)
	for scanner.Scan() {
		mountString := scanner.Text()
		parsedMounts, err := parseMountInfoString(mountString)
		if err != nil {
			return nil, err
		}
		mounts = append(mounts, parsedMounts)
	}

	err := scanner.Err()
	return mounts, err
}

// Parses a mountinfo file line, and converts it to a MountInfo struct.
// An important check here is to see if the hyphen separator, as if it does not exist,
// it means that the line is malformed.
func parseMountInfoString(mountString string) (*MountInfo, error) {
	var err error

	// OptionalFields can be zero, hence these checks to ensure we do not populate the wrong values in the wrong spots
	separatorIndex := strings.Index(mountString, "-")
	if separatorIndex == -1 {
		return nil, fmt.Errorf("no separator found in mountinfo string: %s", mountString)
	}
	beforeFields := strings.Fields(mountString[:separatorIndex])
	afterFields := strings.Fields(mountString[separatorIndex+1:])
	if (len(beforeFields) + len(afterFields)) < 7 {
		return nil, fmt.Errorf("too few fields")
	}

	mount := &MountInfo{
		MajorMinorVer:  getStringSliceElement(beforeFields, 2, ""),
		Root:           getStringSliceElement(beforeFields, 3, ""),
		MountPoint:     getStringSliceElement(beforeFields, 4, ""),
		Options:        mountOptionsParser(getStringSliceElement(beforeFields, 5, "")),
		OptionalFields: nil,
		FSType:         getStringSliceElement(afterFields, 0, ""),
		Source:         getStringSliceElement(afterFields, 1, ""),
		SuperOptions:   mountOptionsParser(getStringSliceElement(afterFields, 2, "")),
	}

	mount.MountId, err = strconv.Atoi(getStringSliceElement(beforeFields, 0, ""))
	if err != nil {
		return nil, fmt.Errorf("failed to parse mount ID")
	}
	mount.ParentId, err = strconv.Atoi(getStringSliceElement(beforeFields, 1, ""))
	if err != nil {
		return nil, fmt.Errorf("failed to parse parent ID")
	}
	// Has optional fields, which is a space separated list of values.
	// Example: shared:2 master:7
	if len(beforeFields) > 6 {
		mount.OptionalFields = make(map[string]string)
		optionalFields := beforeFields[6:]
		for _, field := range optionalFields {
			optionSplit := strings.Split(field, ":")
			target, value := optionSplit[0], ""
			if len(optionSplit) == 2 {
				value = optionSplit[1]
			}
			// Checks if the 'keys' in the optional fields in the mountinfo line are acceptable.
			// Allowed 'keys' are shared, master, propagate_from, unbindable.
			if _, ok := validOptionalFields[target]; ok {
				mount.OptionalFields[target] = value
			}
		}
	}
	return mount, nil
}

// Parses the mount options, superblock options.
func mountOptionsParser(mountOptions string) map[string]string {
	opts := make(map[string]string)
	options := strings.Split(mountOptions, ",")
	for _, opt := range options {
		splitOption := strings.Split(opt, "=")
		if len(splitOption) < 2 {
			key := splitOption[0]
			opts[key] = ""
		} else {
			key, value := splitOption[0], splitOption[1]
			opts[key] = value
		}
	}
	return opts
}

// Retrieves mountinfo information from `/proc/self/mountinfo`.
func GetMounts() ([]*MountInfo, error) {
	f, err := os.Open("/proc/self/mountinfo")
	if err != nil {
		return nil, err
	}
	defer f.Close()
	return parseMountInfo(f)
}

// Retrieves mountinfo information from a processes' `/proc/<pid>/mountinfo`.
func GetProcMounts(pid int) ([]*MountInfo, error) {
	f, err := os.Open(fmt.Sprintf("/proc/%d/mountinfo", pid))
	if err != nil {
		return nil, err
	}
	defer f.Close()
	return parseMountInfo(f)
}
