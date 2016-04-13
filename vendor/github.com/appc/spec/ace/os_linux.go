// Copyright 2015 The appc Authors
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

// +build linux

package main

import (
	"fmt"
	"os"
)

func checkMountImpl(d string, readonly bool) error {
	mountinfoPath := fmt.Sprintf("/proc/self/mountinfo")
	mi, err := os.Open(mountinfoPath)
	if err != nil {
		return err
	}
	defer mi.Close()

	isMounted, ro, err := parseMountinfo(mi, d)
	if err != nil {
		return err
	}
	if !isMounted {
		return fmt.Errorf("%q is not a mount point", d)
	}

	if ro == readonly {
		return nil
	} else {
		return fmt.Errorf("%q mounted ro=%t, want %t", d, ro, readonly)
	}
}
