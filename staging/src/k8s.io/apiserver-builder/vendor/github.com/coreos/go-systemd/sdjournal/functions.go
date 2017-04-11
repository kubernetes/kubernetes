// Copyright 2015 RedHat, Inc.
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

package sdjournal

import (
	"github.com/coreos/pkg/dlopen"
	"sync"
	"unsafe"
)

var (
	// lazy initialized
	libsystemdHandle *dlopen.LibHandle

	libsystemdMutex     = &sync.Mutex{}
	libsystemdFunctions = map[string]unsafe.Pointer{}
	libsystemdNames     = []string{
		// systemd < 209
		"libsystemd-journal.so.0",
		"libsystemd-journal.so",

		// systemd >= 209 merged libsystemd-journal into libsystemd proper
		"libsystemd.so.0",
		"libsystemd.so",
	}
)

func getFunction(name string) (unsafe.Pointer, error) {
	libsystemdMutex.Lock()
	defer libsystemdMutex.Unlock()

	if libsystemdHandle == nil {
		h, err := dlopen.GetHandle(libsystemdNames)
		if err != nil {
			return nil, err
		}

		libsystemdHandle = h
	}

	f, ok := libsystemdFunctions[name]
	if !ok {
		var err error
		f, err = libsystemdHandle.GetSymbolPointer(name)
		if err != nil {
			return nil, err
		}

		libsystemdFunctions[name] = f
	}

	return f, nil
}
