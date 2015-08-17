// Copyright 2015 CoreOS, Inc.
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

package ns

import (
	"fmt"
	"os"
	"runtime"
	"syscall"
)

var setNsMap = map[string]uintptr{
	"386":   346,
	"amd64": 308,
	"arm":   374,
}

// SetNS sets the network namespace on a target file.
func SetNS(f *os.File, flags uintptr) error {
	if runtime.GOOS != "linux" {
		return fmt.Errorf("unsupported OS: %s", runtime.GOOS)
	}

	trap, ok := setNsMap[runtime.GOARCH]
	if !ok {
		return fmt.Errorf("unsupported arch: %s", runtime.GOARCH)
	}

	_, _, err := syscall.RawSyscall(trap, f.Fd(), flags, 0)
	if err != 0 {
		return err
	}

	return nil
}

// WithNetNSPath executes the passed closure under the given network
// namespace, restoring the original namespace afterwards.
// Changing namespaces must be done on a goroutine that has been
// locked to an OS thread. If lockThread arg is true, this function
// locks the goroutine prior to change namespace and unlocks before
// returning
func WithNetNSPath(nspath string, lockThread bool, f func(*os.File) error) error {
	ns, err := os.Open(nspath)
	if err != nil {
		return fmt.Errorf("Failed to open %v: %v", nspath, err)
	}
	defer ns.Close()
	return WithNetNS(ns, lockThread, f)
}

// WithNetNS executes the passed closure under the given network
// namespace, restoring the original namespace afterwards.
// Changing namespaces must be done on a goroutine that has been
// locked to an OS thread. If lockThread arg is true, this function
// locks the goroutine prior to change namespace and unlocks before
// returning
func WithNetNS(ns *os.File, lockThread bool, f func(*os.File) error) error {
	if lockThread {
		runtime.LockOSThread()
		defer runtime.UnlockOSThread()
	}
	// save a handle to current (host) network namespace
	thisNS, err := os.Open("/proc/self/ns/net")
	if err != nil {
		return fmt.Errorf("Failed to open /proc/self/ns/net: %v", err)
	}
	defer thisNS.Close()

	if err = SetNS(ns, syscall.CLONE_NEWNET); err != nil {
		return fmt.Errorf("Error switching to ns %v: %v", ns.Name(), err)
	}

	if err = f(thisNS); err != nil {
		return err
	}

	// switch back
	return SetNS(thisNS, syscall.CLONE_NEWNET)
}
