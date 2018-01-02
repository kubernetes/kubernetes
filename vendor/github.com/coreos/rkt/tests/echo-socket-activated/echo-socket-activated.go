// Copyright 2015 The rkt Authors
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

package main

import (
	"fmt"
	"io"
	"net"
	"os"
	"strconv"

	"github.com/coreos/rkt/pkg/sys"
)

const (
	SD_LISTEN_FDS_START = 3
)

func sdListenFDs(unsetEnvironment bool) (int, error) {
	defer func() {
		if unsetEnvironment {
			os.Unsetenv("LISTEN_PID")
			os.Unsetenv("LISTEN_FDS")
		}
	}()

	e := os.Getenv("LISTEN_PID")
	if e == "" {
		return 0, nil
	}

	pid, err := strconv.Atoi(e)
	if err != nil {
		return -1, err
	}

	if os.Getpid() != pid {
		return 0, nil
	}

	e = os.Getenv("LISTEN_FDS")
	if e == "" {
		return 0, nil
	}

	n, err := strconv.Atoi(e)
	if err != nil {
		return -1, err
	}

	for fd := SD_LISTEN_FDS_START; fd < SD_LISTEN_FDS_START+n; fd++ {
		if err := sys.CloseOnExec(fd, true); err != nil {
			return -1, err
		}
	}

	return n, nil
}

func main() {
	var fd uintptr

	n, err := sdListenFDs(false)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error getting socket-activated FDs: %v\n", err)
		os.Exit(254)
	}
	switch {
	case n == 1:
		fd = SD_LISTEN_FDS_START
	case n > 1:
		fmt.Fprintf(os.Stderr, "Too many file descriptors received.\n")
		os.Exit(254)
	default:
		fmt.Fprintf(os.Stderr, "Not socket activated.\n")
		os.Exit(254)
	}

	f := os.NewFile(fd, "")
	defer f.Close()

	l, err := net.FileListener(f)
	if err != nil {
		fmt.Fprintf(os.Stderr, "FileListener: %v.\n", err)
		os.Exit(254)
	}

	conn, err := l.Accept()
	if err != nil {
		fmt.Fprintf(os.Stderr, "Accept: %v.\n", err)
		os.Exit(254)
	}

	buf := make([]byte, 1024)
	for {
		nr, err := conn.Read(buf)
		if err != nil {
			if err != io.EOF {
				fmt.Fprintf(os.Stderr, "Read: %v.\n", err)
				os.Exit(254)
			}
			break
		}

		data := buf[0:nr]
		if _, err := conn.Write(data); err != nil {
			fmt.Fprintf(os.Stderr, "Write: %v.\n", err)
			os.Exit(254)
		}
	}
}
