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
	"bytes"
	"fmt"
	"strconv"
	"strings"

	"github.com/prometheus/procfs/internal/util"
)

// A MountInfo is a type that describes the details, options
// for each mount, parsed from /proc/self/mountinfo.
// The fields described in each entry of /proc/self/mountinfo
// is described in the following man page.
// http://man7.org/linux/man-pages/man5/proc.5.html
type MountInfo struct {
	// Unique ID for the mount
	MountID int
	// The ID of the parent mount
	ParentID int
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

// Reads each line of the mountinfo file, and returns a list of formatted MountInfo structs.
func parseMountInfo(info []byte) ([]*MountInfo, error) {
	mounts := []*MountInfo{}
	scanner := bufio.NewScanner(bytes.NewReader(info))
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

	mountInfo := strings.Split(mountString, " ")
	mountInfoLength := len(mountInfo)
	if mountInfoLength < 10 {
		return nil, fmt.Errorf("%w: Too few fields in mount string: %s", ErrFileParse, mountString)
	}

	if mountInfo[mountInfoLength-4] != "-" {
		return nil, fmt.Errorf("%w: couldn't find separator in expected field: %s", ErrFileParse, mountInfo[mountInfoLength-4])
	}

	mount := &MountInfo{
		MajorMinorVer:  mountInfo[2],
		Root:           mountInfo[3],
		MountPoint:     mountInfo[4],
		Options:        mountOptionsParser(mountInfo[5]),
		OptionalFields: nil,
		FSType:         mountInfo[mountInfoLength-3],
		Source:         mountInfo[mountInfoLength-2],
		SuperOptions:   mountOptionsParser(mountInfo[mountInfoLength-1]),
	}

	mount.MountID, err = strconv.Atoi(mountInfo[0])
	if err != nil {
		return nil, fmt.Errorf("%w: mount ID: %q", ErrFileParse, mount.MountID)
	}
	mount.ParentID, err = strconv.Atoi(mountInfo[1])
	if err != nil {
		return nil, fmt.Errorf("%w: parent ID: %q", ErrFileParse, mount.ParentID)
	}
	// Has optional fields, which is a space separated list of values.
	// Example: shared:2 master:7
	if mountInfo[6] != "" {
		mount.OptionalFields, err = mountOptionsParseOptionalFields(mountInfo[6 : mountInfoLength-4])
		if err != nil {
			return nil, fmt.Errorf("%w: %w", ErrFileParse, err)
		}
	}
	return mount, nil
}

// mountOptionsIsValidField checks a string against a valid list of optional fields keys.
func mountOptionsIsValidField(s string) bool {
	switch s {
	case
		"shared",
		"master",
		"propagate_from",
		"unbindable":
		return true
	}
	return false
}

// mountOptionsParseOptionalFields parses a list of optional fields strings into a double map of strings.
func mountOptionsParseOptionalFields(o []string) (map[string]string, error) {
	optionalFields := make(map[string]string)
	for _, field := range o {
		optionSplit := strings.SplitN(field, ":", 2)
		value := ""
		if len(optionSplit) == 2 {
			value = optionSplit[1]
		}
		if mountOptionsIsValidField(optionSplit[0]) {
			optionalFields[optionSplit[0]] = value
		}
	}
	return optionalFields, nil
}

// mountOptionsParser parses the mount options, superblock options.
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

// GetMounts retrieves mountinfo information from `/proc/self/mountinfo`.
func GetMounts() ([]*MountInfo, error) {
	data, err := util.ReadFileNoStat("/proc/self/mountinfo")
	if err != nil {
		return nil, err
	}
	return parseMountInfo(data)
}

// GetProcMounts retrieves mountinfo information from a processes' `/proc/<pid>/mountinfo`.
func GetProcMounts(pid int) ([]*MountInfo, error) {
	data, err := util.ReadFileNoStat(fmt.Sprintf("/proc/%d/mountinfo", pid))
	if err != nil {
		return nil, err
	}
	return parseMountInfo(data)
}
