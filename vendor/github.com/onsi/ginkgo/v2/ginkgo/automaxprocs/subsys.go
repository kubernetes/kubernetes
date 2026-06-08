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
	"strconv"
	"strings"
)

const (
	_cgroupSep       = ":"
	_cgroupSubsysSep = ","
)

const (
	_csFieldIDID = iota
	_csFieldIDSubsystems
	_csFieldIDName
	_csFieldCount
)

// CGroupSubsys represents the data structure for entities in
// `/proc/$PID/cgroup`. See also proc(5) for more information.
type CGroupSubsys struct {
	ID         int
	Subsystems []string
	Name       string
}

// NewCGroupSubsysFromLine returns a new *CGroupSubsys by parsing a string in
// the format of `/proc/$PID/cgroup`
func NewCGroupSubsysFromLine(line string) (*CGroupSubsys, error) {
	fields := strings.SplitN(line, _cgroupSep, _csFieldCount)

	if len(fields) != _csFieldCount {
		return nil, cgroupSubsysFormatInvalidError{line}
	}

	id, err := strconv.Atoi(fields[_csFieldIDID])
	if err != nil {
		return nil, err
	}

	cgroup := &CGroupSubsys{
		ID:         id,
		Subsystems: strings.Split(fields[_csFieldIDSubsystems], _cgroupSubsysSep),
		Name:       fields[_csFieldIDName],
	}

	return cgroup, nil
}

// parseCGroupSubsystems parses procPathCGroup (usually at `/proc/$PID/cgroup`)
// and returns a new map[string]*CGroupSubsys.
func parseCGroupSubsystems(procPathCGroup string) (map[string]*CGroupSubsys, error) {
	cgroupFile, err := os.Open(procPathCGroup)
	if err != nil {
		return nil, err
	}
	defer cgroupFile.Close()

	scanner := bufio.NewScanner(cgroupFile)
	subsystems := make(map[string]*CGroupSubsys)

	for scanner.Scan() {
		cgroup, err := NewCGroupSubsysFromLine(scanner.Text())
		if err != nil {
			return nil, err
		}
		for _, subsys := range cgroup.Subsystems {
			subsystems[subsys] = cgroup
		}
	}

	if err := scanner.Err(); err != nil {
		return nil, err
	}

	return subsystems, nil
}
