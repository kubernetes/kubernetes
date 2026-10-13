// Copyright (c) 2017 Uber Technologies, Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

//go:build linux
// +build linux

package automaxprocs

import (
	"bufio"
	"os"
	"path/filepath"
	"strconv"
	"strings"
)

const (
	_mountInfoSep               = " "
	_mountInfoOptsSep           = ","
	_mountInfoOptionalFieldsSep = "-"
)

const (
	_miFieldIDMountID = iota
	_miFieldIDParentID
	_miFieldIDDeviceID
	_miFieldIDRoot
	_miFieldIDMountPoint
	_miFieldIDOptions
	_miFieldIDOptionalFields

	_miFieldCountFirstHalf
)

const (
	_miFieldOffsetFSType = iota
	_miFieldOffsetMountSource
	_miFieldOffsetSuperOptions

	_miFieldCountSecondHalf
)

const _miFieldCountMin = _miFieldCountFirstHalf + _miFieldCountSecondHalf

// MountPoint is the data structure for the mount points in
// `/proc/$PID/mountinfo`. See also proc(5) for more information.
type MountPoint struct {
	MountID        int
	ParentID       int
	DeviceID       string
	Root           string
	MountPoint     string
	Options        []string
	OptionalFields []string
	FSType         string
	MountSource    string
	SuperOptions   []string
}

// NewMountPointFromLine parses a line read from `/proc/$PID/mountinfo` and
// returns a new *MountPoint.
func NewMountPointFromLine(line string) (*MountPoint, error) {
	fields := strings.Split(line, _mountInfoSep)

	if len(fields) < _miFieldCountMin {
		return nil, mountPointFormatInvalidError{line}
	}

	mountID, err := strconv.Atoi(fields[_miFieldIDMountID])
	if err != nil {
		return nil, err
	}

	parentID, err := strconv.Atoi(fields[_miFieldIDParentID])
	if err != nil {
		return nil, err
	}

	for i, field := range fields[_miFieldIDOptionalFields:] {
		if field == _mountInfoOptionalFieldsSep {
			// End of optional fields.
			fsTypeStart := _miFieldIDOptionalFields + i + 1

			// Now we know where the optional fields end, split the line again with a
			// limit to avoid issues with spaces in super options as present on WSL.
			fields = strings.SplitN(line, _mountInfoSep, fsTypeStart+_miFieldCountSecondHalf)
			if len(fields) != fsTypeStart+_miFieldCountSecondHalf {
				return nil, mountPointFormatInvalidError{line}
			}

			miFieldIDFSType := _miFieldOffsetFSType + fsTypeStart
			miFieldIDMountSource := _miFieldOffsetMountSource + fsTypeStart
			miFieldIDSuperOptions := _miFieldOffsetSuperOptions + fsTypeStart

			return &MountPoint{
				MountID:        mountID,
				ParentID:       parentID,
				DeviceID:       fields[_miFieldIDDeviceID],
				Root:           fields[_miFieldIDRoot],
				MountPoint:     fields[_miFieldIDMountPoint],
				Options:        strings.Split(fields[_miFieldIDOptions], _mountInfoOptsSep),
				OptionalFields: fields[_miFieldIDOptionalFields:(fsTypeStart - 1)],
				FSType:         fields[miFieldIDFSType],
				MountSource:    fields[miFieldIDMountSource],
				SuperOptions:   strings.Split(fields[miFieldIDSuperOptions], _mountInfoOptsSep),
			}, nil
		}
	}

	return nil, mountPointFormatInvalidError{line}
}

// Translate converts an absolute path inside the *MountPoint's file system to
// the host file system path in the mount namespace the *MountPoint belongs to.
func (mp *MountPoint) Translate(absPath string) (string, error) {
	relPath, err := filepath.Rel(mp.Root, absPath)

	if err != nil {
		return "", err
	}
	if relPath == ".." || strings.HasPrefix(relPath, "../") {
		return "", pathNotExposedFromMountPointError{
			mountPoint: mp.MountPoint,
			root:       mp.Root,
			path:       absPath,
		}
	}

	return filepath.Join(mp.MountPoint, relPath), nil
}

// parseMountInfo parses procPathMountInfo (usually at `/proc/$PID/mountinfo`)
// and yields parsed *MountPoint into newMountPoint.
func parseMountInfo(procPathMountInfo string, newMountPoint func(*MountPoint) error) error {
	mountInfoFile, err := os.Open(procPathMountInfo)
	if err != nil {
		return err
	}
	defer mountInfoFile.Close()

	scanner := bufio.NewScanner(mountInfoFile)

	for scanner.Scan() {
		mountPoint, err := NewMountPointFromLine(scanner.Text())
		if err != nil {
			return err
		}
		if err := newMountPoint(mountPoint); err != nil {
			return err
		}
	}

	return scanner.Err()
}
