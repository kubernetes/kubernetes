/*
Copyright 2013 CoreOS Inc.

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

// Package activation implements primitives for systemd socket activation.
package activation

import (
	"os"
	"strconv"
	"syscall"
)

// based on: https://gist.github.com/alberts/4640792
const (
	listenFdsStart = 3
)

func Files(unsetEnv bool) []*os.File {
	if unsetEnv {
		// there is no way to unset env in golang os package for now
		// https://code.google.com/p/go/issues/detail?id=6423
		defer os.Setenv("LISTEN_PID", "")
		defer os.Setenv("LISTEN_FDS", "")
	}

	pid, err := strconv.Atoi(os.Getenv("LISTEN_PID"))
	if err != nil || pid != os.Getpid() {
		return nil
	}

	nfds, err := strconv.Atoi(os.Getenv("LISTEN_FDS"))
	if err != nil || nfds == 0 {
		return nil
	}

	var files []*os.File
	for fd := listenFdsStart; fd < listenFdsStart+nfds; fd++ {
		syscall.CloseOnExec(fd)
		files = append(files, os.NewFile(uintptr(fd), "LISTEN_FD_"+strconv.Itoa(fd)))
	}

	return files
}
