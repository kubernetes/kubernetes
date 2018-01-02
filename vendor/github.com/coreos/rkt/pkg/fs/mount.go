// Copyright 2016 The rkt Authors
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

package fs

// Mounter is the interface that mounts a filesystem. It's signature is the same as syscall.Mount.
type Mounter interface {
	Mount(source string, target string, fstype string, flags uintptr, data string) error
}

// MounterFunc is a functional type that implements Mounter.
// The mount syscall can be wrapped using MounterFunc(syscall.Mount) to be used as a Mounter.
type MounterFunc func(string, string, string, uintptr, string) error

func (m MounterFunc) Mount(source string, target string, fstype string, flags uintptr, data string) error {
	return m(source, target, fstype, flags, data)
}

// Unmounter is the interface that unmounts a filesystem. It's signature is the same as syscall.Unmount.
type Unmounter interface {
	Unmount(target string, flags int) error
}

// UnmounterFunc is a functional type that implements Unmounter.
// The unmount syscall can be wrapped using UnmounterFunc(syscall.Unmount) to be used as an Unmounter.
type UnmounterFunc func(string, int) error

func (m UnmounterFunc) Unmount(target string, flags int) error {
	return m(target, flags)
}

// A MountUnmounter is the interface that mounts, and unmounts filesystems.
type MountUnmounter interface {
	Mounter
	Unmounter
}

type loggingMounter struct {
	m    Mounter
	um   Unmounter
	logf func(string, ...interface{})
}

// NewLoggingMounter returns a MountUnmounter that logs mount events using the given logger func.
func NewLoggingMounter(m Mounter, um Unmounter, logf func(string, ...interface{})) MountUnmounter {
	return &loggingMounter{m, um, logf}
}

func (l *loggingMounter) Mount(source string, target string, fstype string, flags uintptr, data string) error {
	l.logf(
		"mounting source %q target %q fstype %q flags %v data %q",
		source, target, fstype, mountFlags(flags), data,
	)

	return l.m.Mount(source, target, fstype, flags, data)
}

func (l *loggingMounter) Unmount(target string, flags int) error {
	l.logf(
		"unmounting target %q flags %v",
		target, mountFlags(flags),
	)

	return l.um.Unmount(target, flags)
}

// mountFlags wraps uintptr mount flags.
// It is supposed to implement the String() method for log output.
// The implementation is system dependent, gated by build tags.
type mountFlags uintptr
