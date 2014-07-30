// Copyright 2014 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
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
	"strconv"

	"github.com/google/cadvisor/utils/fs"
)

type ProcessSchedStat struct {
	// Number of processes
	NumProcesses int

	// Total time spent on the cpu (Unit: jiffy)
	Running uint64

	// Total time spent waiting on a runqueue (Unit: jiffy)
	RunWait uint64

	// # of timeslices run on this cpu (Unit: jiffy)
	NumTimeSlices uint64
}

func readUint64List(r io.Reader) ([]uint64, error) {
	ret := make([]uint64, 0, 4)
	scanner := bufio.NewScanner(r)
	scanner.Split(bufio.ScanWords)

	for scanner.Scan() {
		str := scanner.Text()
		u, err := strconv.ParseUint(str, 10, 64)
		if err != nil {
			return nil, err
		}
		ret = append(ret, u)
	}

	if err := scanner.Err(); err != nil {
		return nil, err
	}

	return ret, nil
}

// Add() read the schedstat of pid, and add stat to the fields
// in self parameters. This function is useful if one wants to read stats of
// a group of processes.
func (self *ProcessSchedStat) Add(pid int) error {
	if self == nil {
		return fmt.Errorf("nil stat")
	}

	path := fmt.Sprintf("/proc/%d/schedstat", pid)
	f, err := fs.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()
	v, err := readUint64List(f)
	if err != nil {
		return err
	}
	if len(v) < 3 {
		return fmt.Errorf("only %v fields read from %v: %v", len(v), path, v)
	}
	self.Running += v[0]
	self.RunWait += v[1]
	self.NumTimeSlices += v[2]
	self.NumProcesses++
	return nil
}
