/*
   Copyright The containerd Authors.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/

package cgroup1

import (
	"os"

	v1 "github.com/containerd/cgroups/v3/cgroup1/stats"
	specs "github.com/opencontainers/runtime-spec/specs-go"
)

type procType = string

const (
	cgroupProcs    procType = "cgroup.procs"
	cgroupTasks    procType = "tasks"
	defaultDirPerm          = 0o755
)

// defaultFilePerm is a var so that the test framework can change the filemode
// of all files created when the tests are running.  The difference between the
// tests and real world use is that files like "cgroup.procs" will exist when writing
// to a read cgroup filesystem and do not exist prior when running in the tests.
// this is set to a non 0 value in the test code
var defaultFilePerm = os.FileMode(0)

type Process struct {
	// Subsystem is the name of the subsystem that the process / task is in.
	Subsystem Name
	// Pid is the process id of the process / task.
	Pid int
	// Path is the full path of the subsystem and location that the process / task is in.
	Path string
}

type Task = Process

// Cgroup handles interactions with the individual groups to perform
// actions on them as them main interface to this cgroup package
type Cgroup interface {
	// New creates a new cgroup under the calling cgroup
	New(string, *specs.LinuxResources) (Cgroup, error)
	// Add adds a process to the cgroup (cgroup.procs). Without additional arguments,
	// the process is added to all the cgroup subsystems. When giving Add a list of
	// subsystem names, the process is only added to those subsystems, provided that
	// they are active in the targeted cgroup.
	Add(Process, ...Name) error
	// AddProc adds the process with the given id to the cgroup (cgroup.procs).
	// Without additional arguments, the process with the given id is added to all
	// the cgroup subsystems. When giving AddProc a list of subsystem names, the process
	// id is only added to those subsystems, provided that they are active in the targeted
	// cgroup.
	AddProc(uint64, ...Name) error
	// AddTask adds a process to the cgroup (tasks). Without additional arguments, the
	// task is added to all the cgroup subsystems. When giving AddTask a list of subsystem
	// names, the task is only added to those subsystems, provided that they are active in
	// the targeted cgroup.
	AddTask(Process, ...Name) error
	// Delete removes the cgroup as a whole
	Delete() error
	// MoveTo moves all the processes under the calling cgroup to the provided one
	// subsystems are moved one at a time
	MoveTo(Cgroup) error
	// Stat returns the stats for all subsystems in the cgroup
	Stat(...ErrorHandler) (*v1.Metrics, error)
	// Update updates all the subsystems with the provided resource changes
	Update(resources *specs.LinuxResources) error
	// Processes returns all the processes in a select subsystem for the cgroup
	Processes(Name, bool) ([]Process, error)
	// Tasks returns all the tasks in a select subsystem for the cgroup
	Tasks(Name, bool) ([]Task, error)
	// Freeze freezes or pauses all processes inside the cgroup
	Freeze() error
	// Thaw thaw or resumes all processes inside the cgroup
	Thaw() error
	// OOMEventFD returns the memory subsystem's event fd for OOM events
	OOMEventFD() (uintptr, error)
	// RegisterMemoryEvent returns the memory subsystems event fd for whatever memory event was
	// registered for. Can alternatively register for the oom event with this method.
	RegisterMemoryEvent(MemoryEvent) (uintptr, error)
	// State returns the cgroups current state
	State() State
	// Subsystems returns all the subsystems in the cgroup
	Subsystems() []Subsystem
}
