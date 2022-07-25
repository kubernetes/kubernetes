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

	"golang.org/x/sys/unix"
)

// MaxSendfdLen is the maximum length of the name of a file descriptor being
// sent using SendFd. The name of the file handle returned by RecvFd will never
// be larger than this value.
const MaxNameLen = 4096

// oobSpace is the size of the oob slice required to store a single FD. Note
// that unix.UnixRights appears to make the assumption that fd is always int32,
// so sizeof(fd) = 4.
var oobSpace = unix.CmsgSpace(4)

// RecvFd waits for a file descriptor to be sent over the given AF_UNIX
// socket. The file name of the remote file descriptor will be recreated
// locally (it is sent as non-auxiliary data in the same payload).
func RecvFd(socket *os.File) (*os.File, error) {
	// For some reason, unix.Recvmsg uses the length rather than the capacity
	// when passing the msg_controllen and other attributes to recvmsg.  So we
	// have to actually set the length.
	name := make([]byte, MaxNameLen)
	oob := make([]byte, oobSpace)

	sockfd := socket.Fd()
	n, oobn, _, _, err := unix.Recvmsg(int(sockfd), name, oob, 0)
	if err != nil {
		return nil, err
	}

	if n >= MaxNameLen || oobn != oobSpace {
		return nil, fmt.Errorf("recvfd: incorrect number of bytes read (n=%d oobn=%d)", n, oobn)
	}

	// Truncate.
	name = name[:n]
	oob = oob[:oobn]

	scms, err := unix.ParseSocketControlMessage(oob)
	if err != nil {
		return nil, err
	}
	if len(scms) != 1 {
		return nil, fmt.Errorf("recvfd: number of SCMs is not 1: %d", len(scms))
	}
	scm := scms[0]

	fds, err := unix.ParseUnixRights(&scm)
	if err != nil {
		return nil, err
	}
	if len(fds) != 1 {
		return nil, fmt.Errorf("recvfd: number of fds is not 1: %d", len(fds))
	}
	fd := uintptr(fds[0])

	return os.NewFile(fd, string(name)), nil
}

// SendFd sends a file descriptor over the given AF_UNIX socket. In
// addition, the file.Name() of the given file will also be sent as
// non-auxiliary data in the same payload (allowing to send contextual
// information for a file descriptor).
func SendFd(socket *os.File, name string, fd uintptr) error {
	if len(name) >= MaxNameLen {
		return fmt.Errorf("sendfd: filename too long: %s", name)
	}
	return SendFds(socket, []byte(name), int(fd))
}

// SendFds sends a list of files descriptor and msg over the given AF_UNIX socket.
func SendFds(socket *os.File, msg []byte, fds ...int) error {
	oob := unix.UnixRights(fds...)
	return unix.Sendmsg(int(socket.Fd()), msg, oob, nil, 0)
}
