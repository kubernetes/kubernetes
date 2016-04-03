// Copyright 2015 The go4 Authors
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

// +build linux netbsd openbsd dragonfly nacl

package osutil

import (
	"errors"
	"os"
	"path/filepath"
	"runtime"
)

func executable() (string, error) {
	var procfn string
	switch runtime.GOOS {
	default:
		return "", errors.New("Executable not implemented for " + runtime.GOOS)
	case "linux":
		procfn = "/proc/self/exe"
	case "netbsd":
		procfn = "/proc/curproc/exe"
	case "openbsd":
		procfn = "/proc/curproc/file"
	case "dragonfly":
		procfn = "/proc/curproc/file"
	}
	p, err := os.Readlink(procfn)
	return filepath.Clean(p), err
}
