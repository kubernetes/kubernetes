package utils

/*
 * Copyright 2016, 2017 SUSE LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import (
	"fmt"
	"os"
	"runtime"

	"golang.org/x/sys/unix"
)

// MaxNameLen is the maximum length of the name of a file descriptor being sent
// using SendFile. The name of the file handle returned by RecvFile will never be
// larger than this value.
const MaxNameLen = 4096

// oobSpace is the size of the oob slice required to store a single FD. Note
// that unix.UnixRights appears to make the assumption that fd is always int32,
// so sizeof(fd) = 4.
var oobSpace = unix.CmsgSpace(4)

// RecvFile waits for a file descriptor to be sent over the given AF_UNIX
// socket. The file name of the remote file descriptor will be recreated
// locally (it is sent as non-auxiliary data in the same payload).
func RecvFile(socket *os.File) (_ *os.File, Err error) {
	name := make([]byte, MaxNameLen)
	oob := make([]byte, oobSpace)

	sockfd := socket.Fd()
	n, oobn, _, _, err := unix.Recvmsg(int(sockfd), name, oob, unix.MSG_CMSG_CLOEXEC)
	if err != nil {
		return nil, err
	}
	if n >= MaxNameLen || oobn != oobSpace {
		return nil, fmt.Errorf("recvfile: incorrect number of bytes read (n=%d oobn=%d)", n, oobn)
	}
	// Truncate.
	name = name[:n]
	oob = oob[:oobn]

	scms, err := unix.ParseSocketControlMessage(oob)
	if err != nil {
		return nil, err
	}

	// We cannot control how many SCM_RIGHTS we receive, and upon receiving
	// them all of the descriptors are installed in our fd table, so we need to
	// parse all of the SCM_RIGHTS we received in order to close all of the
	// descriptors on error.
	var fds []int
	defer func() {
		for i, fd := range fds {
			if i == 0 && Err == nil {
				// Only close the first one on error.
				continue
			}
			// Always close extra ones.
			_ = unix.Close(fd)
		}
	}()
	var lastErr error
	for _, scm := range scms {
		if scm.Header.Type == unix.SCM_RIGHTS {
			scmFds, err := unix.ParseUnixRights(&scm)
			if err != nil {
				lastErr = err
			} else {
				fds = append(fds, scmFds...)
			}
		}
	}
	if lastErr != nil {
		return nil, lastErr
	}

	// We do this after collecting the fds to make sure we close them all when
	// returning an error here.
	if len(scms) != 1 {
		return nil, fmt.Errorf("recvfd: number of SCMs is not 1: %d", len(scms))
	}
	if len(fds) != 1 {
		return nil, fmt.Errorf("recvfd: number of fds is not 1: %d", len(fds))
	}
	return os.NewFile(uintptr(fds[0]), string(name)), nil
}

// SendFile sends a file over the given AF_UNIX socket. file.Name() is also
// included so that if the other end uses RecvFile, the file will have the same
// name information.
func SendFile(socket *os.File, file *os.File) error {
	name := file.Name()
	if len(name) >= MaxNameLen {
		return fmt.Errorf("sendfd: filename too long: %s", name)
	}
	err := SendRawFd(socket, name, file.Fd())
	runtime.KeepAlive(file)
	return err
}

// SendRawFd sends a specific file descriptor over the given AF_UNIX socket.
func SendRawFd(socket *os.File, msg string, fd uintptr) error {
	oob := unix.UnixRights(int(fd))
	return unix.Sendmsg(int(socket.Fd()), []byte(msg), oob, nil, 0)
}
